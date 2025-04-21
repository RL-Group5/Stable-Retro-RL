import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# {'lr': 0.00036273949280554415, 'clr': 0.00027539943527406523, 'eps': 0.2978548811971866, 'lreps': 1.0395397543281306e-07, 'gamma': 0.9328669115748676, 'steps_batch': 824, 'updates_per_iteration': 6, 'n_envs': 4, 'frame_skip': 4, 'stickprob': 0.4932402028812446}

class CNNPolicy(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNPolicy, self).__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy_input).view(1, -1).size(1)

        self.actor_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.GELU(),
            nn.Linear(512, num_actions)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )

    def forward_conv(self, x):
        x = x / 255.0
        return self.conv(x)

    def actor(self, x):
        x = self.forward_conv(x)
        x = torch.flatten(x, 1)
        return self.actor_head(x)

    def critic(self, x):
        x = self.forward_conv(x)
        x = torch.flatten(x, 1)
        return self.critic_head(x)


class PPO:
    def __init__(self, env, **hyperparameters):
        self.callback = hyperparameters.pop("callback", None)
        self.env = env
        self._init_hyperparameters(hyperparameters)
        self.num_timesteps = 0
        obs_shape = env.observation_space.shape
        act_dim = env.action_space.n

        self.policy = CNNPolicy(obs_shape, act_dim)
        self.actor_optimizer = torch.optim.AdamW(self.policy.actor_head.parameters(), lr=self.lr, eps=self.lreps)
        self.critic_optimizer = torch.optim.AdamW(self.policy.critic_head.parameters(), lr=self.clr, eps=self.lreps)

        self.reward_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def _init_hyperparameters(self, hyperparameters={}):
        self.steps_batch = hyperparameters.get("steps_batch", 2000)
        self.steps_episode = hyperparameters.get("steps_episode", 2000)
        self.gamma = hyperparameters.get("gamma", 0.9328669115748676)
        self.updates_per_iteration = hyperparameters.get("updates_per_iteration", 6)
        self.lr = hyperparameters.get("lr", 0.00036273949280554415)
        self.clr = hyperparameters.get("clr", 0.00027539943527406523)
        self.eps = hyperparameters.get("eps", 0.2978548811971866)
        self.lreps = hyperparameters.get("lreps", 1.0395397543281306e-07)

    def episodes(self):
        batch_obs = []
        batch_acts = []
        batch_prob = []
        batch_rewards = []
        batch_lens = []

        total_reward = 0
        t_step = 0

        while t_step < self.steps_batch:
            ep_rewards = []

            # Reset environment and extract observation
            reset_result = self.env.reset()
            obs_np = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            # Convert to tensor and ensure batch dim
            obs = torch.tensor(obs_np, dtype=torch.float32)
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)  # -> (batch, C, H, W)

                                      # Expect (n_envs, C, H, W)
            done = False

            # Collect experience for one episode
            for ep in range(self.steps_episode):
                old_obs = obs.clone()
                acts, probs = self.get_action(old_obs)

                # For vectorized envs, convert tensor of actions to numpy array
                if isinstance(acts, torch.Tensor):
                    action = acts.detach().cpu().numpy()
                else:
                    action = np.array([int(acts)])
                
                self.num_timesteps += 1
                if self.callback:
                    self.callback.model = self
                    self.callback.num_timesteps = self.num_timesteps
                    self.callback._on_step()
                # Step through the vectorized env
                step_result = self.env.step(action)
                obs_np, reward = step_result[0], step_result[1]
                done = step_result[2] if len(step_result) > 2 else False

                if isinstance(done, (list, np.ndarray)):
                    done = bool(np.any(done))

                # Convert next obs to tensor and ensure batch dim
                obs = torch.tensor(obs_np, dtype=torch.float32)
                if obs.dim() == 3:
                    obs = obs.unsqueeze(0)

                # Store transition
                batch_obs.append(old_obs)
                batch_acts.append(acts.detach().long())
                batch_prob.append(probs)
                ep_rewards.append(reward)

                total_reward += np.sum(reward) if isinstance(reward, (list, np.ndarray)) else reward
                t_step += 1


            batch_lens.append(ep + 1)
            batch_rewards.append(ep_rewards)

        # Prepare batches
        batch_obs = torch.cat(batch_obs, dim=0)
        batch_acts = torch.cat(batch_acts, dim=0)
        batch_prob = torch.cat(batch_prob, dim=0)
        batch_returns = self.compute_returns(batch_rewards)
        self.reward_history.append(total_reward / len(batch_lens))
        return batch_obs, batch_acts, batch_prob, batch_returns, batch_lens


    def learn(self, steps):
        cur_steps = 0
        iteration = 0
        while cur_steps < steps:
            batch_obs, batch_acts, batch_prob, batch_returns, batch_lens = self.episodes()
            batch_returns = batch_returns.flatten() 
            cur_steps += sum(batch_lens)
            iteration += 1

            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_returns - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.updates_per_iteration):
                V, cur_prob = self.evaluate(batch_obs, batch_acts)
                ratio = torch.exp(cur_prob - batch_prob)

                surr1 = ratio * A_k
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(V, batch_returns)

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            self.actor_loss_history.append(actor_loss.item())
            self.critic_loss_history.append(critic_loss.item())
            avg_len = sum(batch_lens) / len(batch_lens)
            print(f"Iter {iteration} | Steps {cur_steps} | AvgEpLen {avg_len:.2f} | "
                  f"Reward {self.reward_history[-1]:.2f} | "
                  f"ActorLoss {actor_loss:.4f} | CriticLoss {critic_loss:.4f}")
 
    def get_action(self, obs):
        logits = self.policy.actor(obs)
        dist   = torch.distributions.Categorical(logits=logits)
        acts   = dist.sample()              # shape [batch_size]
        probs  = dist.log_prob(acts)        # shape [batch_size]
        return acts, probs.detach()

    def evaluate(self, batch_obs, batch_acts):
        V = self.policy.critic(batch_obs).squeeze()
        logits = self.policy.actor(batch_obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(batch_acts)
        return V, log_prob

    def compute_returns(self, batch_rewards):
        # batch_rewards is a list of lists/arrays, one per episode
        all_returns = []
        for ep in batch_rewards:
            discounted = 0.0
            ep_returns = []
            # ep is a list or 1‑D array of shape (n_envs,)
            for r in reversed(ep):
                discounted = r + self.gamma * discounted
                ep_returns.append(discounted)
            all_returns.append(ep_returns[::-1])  # reverse back to forward order

        # all_returns is list of lists of arrays; stack into one 2D array
        stacked = np.stack(all_returns, axis=0)      # shape (n_eps, ep_len)
        flat = stacked.reshape(-1)                   # shape (n_eps*ep_len,)
        return torch.from_numpy(flat.astype(np.float32))
    
    def predict(self, obs: np.ndarray):
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        acts, _ = self.get_action(obs_t)
        return acts.detach().cpu().numpy(), None
    
    def save_model(self, path="ppo_retro.pth"):
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}", flush=True)
    save = save_model

    def load_model(self, path="ppo_retro.pth"):
        self.policy.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def plot_results(self):
        fig, axs = plt.subplots(2, figsize=(10, 6))
        axs[0].plot(self.reward_history, label="Reward")
        axs[0].set_title("Rewards")
        axs[1].plot(self.actor_loss_history, label="Actor Loss", color="red")
        axs[1].plot(self.critic_loss_history, label="Critic Loss", color="green")
        axs[1].set_title("Losses")
        for ax in axs: ax.legend()
        plt.tight_layout()
        plt.show()
