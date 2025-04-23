import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import os
##{'manager_lr': 0.0001467550968120156, 'worker_lr': 0.00036729931252875995, 'clip_ratio': 0.2366066099777788, 'gamma': 0.9517816171672592, 'gae_lambda': 0.8167624732003352, 'option_duration': 6, 'steps_per_epoch': 512, 'train_epochs': 9, 'n_envs': 2, 'frame_skip': 8, 'stickprob': 0.08746326782814651}
from torch.utils.tensorboard import SummaryWriter

# Define action groups for Mortal Kombat II
ACTION_GROUPS = {
    'MOVEMENT': [0, 1, 2, 3],  # LEFT, RIGHT, UP, DOWN
    'BASIC_ATTACKS': [4, 5, 6],  # PUNCH, KICK, BLOCK
    'SPECIAL_MOVES': [7, 8, 9],  # SPECIAL1, SPECIAL2, SPECIAL3
    'COMBOS': [10, 11, 12]  # COMBO1, COMBO2, COMBO3
}

class ManagerNetwork(nn.Module):
    """High-level policy network that selects action groups/strategies"""
    def __init__(self, input_shape: Tuple[int, ...], num_options: int):
        super().__init__()
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GELU(),
        )

        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            self.conv_out_size = int(np.prod(conv_out.shape[1:]))
            print(f"Manager conv output shape: {conv_out.shape}")
            print(f"Manager conv output size: {self.conv_out_size}")

        # Option selection head
        self.option_head = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_options)
        )

        # Option value head
        self.value_head = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward_conv(self, x):
        x = x / 255.0
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.forward_conv(x)
        x = torch.flatten(x, 1)  # Flatten while preserving batch dim
        option_logits = self.option_head(x)
        value = self.value_head(x)
        return option_logits, value

    def select_option(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select high-level options using the manager network"""
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)  # Add batch dimension if needed
            option_logits, _ = self.forward(obs)
            option_probs = F.softmax(option_logits, dim=-1)
            option_dist = torch.distributions.Categorical(probs=option_probs)
            options = option_dist.sample()
            log_probs = option_dist.log_prob(options)
        return options, log_probs

class WorkerNetwork(nn.Module):
    """Low-level policy network that executes specific actions within an option"""
    def __init__(self, input_shape: Tuple[int, ...], action_dim: int):
        super().__init__()
        c, h, w = input_shape
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GELU(),
        )

        # Calculate conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            self.conv_out_size = int(np.prod(conv_out.shape[1:]))
            print(f"Worker conv output shape: {conv_out.shape}")
            print(f"Worker conv output size: {self.conv_out_size}")

        # Action selection head
        self.action_head = nn.Sequential(
            nn.Linear(self.conv_out_size + 1, 512),  # +1 for option embedding
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        # State value head
        self.value_head = nn.Sequential(
            nn.Linear(self.conv_out_size + 1, 512),  # +1 for option embedding
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward_conv(self, x):
        x = x / 255.0
        return self.conv(x)

    def forward(self, x: torch.Tensor, option_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.forward_conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten to (batch_size, features)
        # Ensure option_embedding is the right shape (batch_size, 1)
        if option_embedding.dim() == 1:
            option_embedding = option_embedding.unsqueeze(1)
        elif option_embedding.dim() > 2:
            option_embedding = option_embedding.view(option_embedding.size(0), -1)
        # Concatenate features with option embedding
        combined = torch.cat([x, option_embedding], dim=1)
        action_logits = self.action_head(combined)
        value = self.value_head(combined)
        return action_logits, value

    def select_action(self, obs: torch.Tensor, options: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select low-level actions using the worker network"""
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)  # Add batch dimension just cause why not
            option_embeddings = options.view(-1, 1).float()  # Ensure correct shape
            action_logits, _ = self.forward(obs, option_embeddings)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
        return actions, log_probs

class HPPO:
    """Hierarchical PPO implementation with fixed defaults, GAE, advantage normalization, entropy bonus"""
    def __init__(self, env, device="cuda",writer=None,**kwargs):
        
        self.env = env
        self.device = torch.device(device)
        self.writer = writer or SummaryWriter(log_dir="./tb_logs/hppo_default")
        self._init_hyperparameters(kwargs)
        
        obs_shape = env.observation_space.shape
        self.num_options = len(ACTION_GROUPS)
        self.action_dim  = env.action_space.n
        self.n_envs      = getattr(env, "num_envs", 1)

        # networks
        self.manager = ManagerNetwork(obs_shape, self.num_options).to(self.device)
        self.worker  = WorkerNetwork(obs_shape, self.action_dim).to(self.device)

        # optimizers
        self.manager_optimizer = torch.optim.AdamW(
            self.manager.parameters(),
            lr=self.manager_lr,
            eps=self.adam_eps
        )
        self.worker_optimizer  = torch.optim.AdamW(
            self.worker.parameters(),
            lr=self.worker_lr,
            eps=self.adam_eps
        )

        # logs
        self.reward_history       = []
        self.manager_loss_history = []
        self.worker_loss_history  = []
        self.num_timesteps        = 0
        
    def _init_hyperparameters(self, hp: Dict):
        # sensible defaults
        self.manager_lr      = hp.get("manager_lr",    3e-4)
        self.worker_lr       = hp.get("worker_lr",     3e-4)
        self.adam_eps        = hp.get("adam_eps",      1e-8)     
        self.gamma           = hp.get("gamma",         0.98)
        self.gae_lambda      = hp.get("gae_lambda",    0.98)
        self.clip_ratio      = hp.get("clip_ratio",    0.2)     
        self.option_duration = hp.get("option_duration", 6)
        self.steps_per_epoch = hp.get("steps_per_epoch", 2048)
        self.train_epochs    = hp.get("train_epochs",    10)
        self.entropy_coef    = hp.get("entropy_coef",  0.01)

    def learn(self, total_steps: int):
        epochs = total_steps // self.steps_per_epoch
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            obs = reset_out[0]
        else:
            obs = reset_out

        current_options = None
        option_steps    = 0
        episode_rewards = np.zeros(self.n_envs)

        for epoch in range(1, epochs+1):
            # buffers
            buf_obs, buf_opts, buf_acts = [], [], []
            buf_opt_lp, buf_act_lp      = [], []
            buf_rews, buf_dones         = [], []
            buf_lens                     = []

            # collect rollout
            for _ in range(self.steps_per_epoch):
                self.num_timesteps += self.n_envs
                # prepare obs tensor
                obs_t = torch.FloatTensor(obs).to(self.device)
                if obs_t.dim() == 4:  # flatten stacked frames
                    N, C, H, W = obs_t.shape
                    obs_t = obs_t.view(N, -1, H, W)

                # choose or continue option
                if current_options is None or option_steps >= self.option_duration:
                    current_options, opt_logp = self.manager.select_option(obs_t)
                    option_steps = 0
                else:
                    opt_logp = torch.zeros(self.n_envs, device=self.device)

                # choose action
                actions, act_logp = self.worker.select_action(obs_t, current_options)

                # --- STEP (handle 4‑ or 5‑tuple returns) ---
                step_out = self.env.step(actions.cpu().numpy())
                if len(step_out) == 4:
                    next_obs, rewards, dones, infos = step_out
                else:  # length 5
                    next_obs, rewards, terminated, truncated, infos = step_out
                    dones = np.logical_or(terminated, truncated)

                # record transition
                buf_obs.append(obs)
                buf_opts.append(current_options.cpu().numpy())
                buf_acts.append(actions.cpu().numpy())
                buf_opt_lp.append(opt_logp.cpu().numpy())
                buf_act_lp.append(act_logp.cpu().numpy())
                buf_rews.append(rewards)
                buf_dones.append(dones)

                # track rewards & resets
                episode_rewards += rewards
                option_steps    += 1
                obs = next_obs

                for i, done in enumerate(dones):
                    if done:
                        buf_lens.append(episode_rewards[i])
                        self.reward_history.append(episode_rewards[i])
                        episode_rewards[i] = 0

                if all(dones):
                    current_options = None
                    reset_out = self.env.reset()
                    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

            # convert buffers
            obs_buf   = torch.FloatTensor(np.array(buf_obs)).to(self.device)
            rew_buf   = torch.FloatTensor(np.array(buf_rews)).to(self.device)
            done_buf  = torch.FloatTensor(np.array(buf_dones)).to(self.device)
            opt_buf   = torch.LongTensor(np.array(buf_opts)).to(self.device)
            act_buf   = torch.LongTensor(np.array(buf_acts)).to(self.device)
            opt_lp_old= torch.FloatTensor(np.array(buf_opt_lp)).to(self.device)
            act_lp_old= torch.FloatTensor(np.array(buf_act_lp)).to(self.device)

            # flatten obs from (T,N,C,H,W) → (TN, C', H, W)
            if obs_buf.dim()==5:
                T,N,C,H,W = obs_buf.shape
                obs_buf = obs_buf.view(T*N, -1, H, W)

            # --- manager update ---
            adv_m, ret_m = self._compute_advantages(
                rew_buf, done_buf, kind="manager", obs=obs_buf, options=None, actions=None
            )
            self._ppo_update(
                self.manager, self.manager_optimizer,
                obs_buf, opt_buf.view(-1), opt_lp_old.view(-1),
                adv_m, ret_m, self.manager_loss_history
            )

            # --- worker update ---
            adv_w, ret_w = self._compute_advantages(
                rew_buf, done_buf, kind="worker",
                obs=obs_buf, options=opt_buf.view(-1,1), actions=act_buf.view(-1)
            )
            self._ppo_update(
                self.worker, self.worker_optimizer,
                obs_buf, act_buf.view(-1), act_lp_old.view(-1),
                adv_w, ret_w, self.worker_loss_history,
                extra_inputs=(opt_buf.view(-1,1),)
            )

            # log
            avg_rew = np.mean(self.reward_history[-len(buf_lens):]) if buf_lens else 0
            self.writer.add_scalar("Reward/Average", avg_rew, self.num_timesteps)
            self.writer.add_scalar("Loss/Manager", self.manager_loss_history[-1], self.num_timesteps)
            self.writer.add_scalar("Loss/Worker", self.worker_loss_history[-1], self.num_timesteps)


            print(f"Epoch {epoch}/{epochs}"
                  f" | Steps {self.num_timesteps}"
                  f" | AvgRew {avg_rew:.2f}"
                  f" | MgrLoss {self.manager_loss_history[-1]:.4f}"
                  f" | WkLoss {self.worker_loss_history[-1]:.4f}")
            self.writer.flush()
        if epoch % 2 == 0:
                print("saving gang")
                self.save(f"./checkpoints_hppo/model_{self.num_timesteps}.pt")
        self.writer.close()
    def _compute_advantages(self, rewards, dones, kind, obs, options=None, actions=None):
        # values from forward pass
        with torch.no_grad():
            if kind=="manager":
                _, vals = self.manager(obs)
            else:
                vals = self.worker(obs, options)[1]
        vals = vals.view(-1).cpu().numpy()
        rews = rewards.cpu().numpy()
        dones = dones.cpu().numpy()
        T, N = rews.shape
        vals = vals.reshape(T, N)

        advs = np.zeros((T, N), dtype=np.float32)
        last = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            next_v = 0 if t==T-1 else vals[t+1]
            delta = rews[t] + self.gamma*next_v*mask - vals[t]
            last = delta + self.gamma*self.gae_lambda*mask*last
            advs[t] = last

        flat_adv = torch.FloatTensor(advs.reshape(-1)).to(self.device)
        norm_adv = (flat_adv - flat_adv.mean())/(flat_adv.std()+1e-8)
        flat_ret = norm_adv + torch.FloatTensor(vals.reshape(-1)).to(self.device)
        return norm_adv, flat_ret

    def _ppo_update(self, network, optimizer, obs, acts, old_logp, adv, ret, loss_hist, extra_inputs=()):
        for _ in range(self.train_epochs):
            if extra_inputs:
                logits, vals = network(obs, *extra_inputs)
            else:
                logits, vals = network(obs)
            vals = vals.view(-1)
            logp_all = F.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, acts.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(logp - old_logp)
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
            pol_loss = -torch.min(ratio*adv, clip_adv).mean()
            val_loss = F.mse_loss(vals, ret)
            entropy = -(logp_all * logp_all.exp()).sum(dim=1).mean()

            loss = pol_loss + 0.5*val_loss - self.entropy_coef*entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_hist.append(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'manager_state_dict': self.manager.state_dict(),
            'worker_state_dict':  self.worker.state_dict(),
            'manager_optimizer':  self.manager_optimizer.state_dict(),
            'worker_optimizer':   self.worker_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path)
        self.manager.load_state_dict(ckpt['manager_state_dict'])
        self.worker.load_state_dict(ckpt['worker_state_dict'])
        self.manager_optimizer.load_state_dict(ckpt['manager_optimizer'])
        self.worker_optimizer.load_state_dict(ckpt['worker_optimizer'])
        print(f"Model loaded from {path}")
