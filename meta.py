import os
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter

class Meta:
    def __init__(
        self,
        base_ppo_cls,
        task_sampler,
        device="cuda",
        meta_lr=1e-3,
        inner_lr=1e-4,
        inner_steps=1,
        meta_batch_size=5,
        **ppo_kwargs
    ):
        """
        Args:
            base_ppo_cls: class for PPO agent
            task_sampler: fn returning a new VecEnv
            device: torch device string, "cpu" or "cuda"
            meta_lr: outer learning rate
            inner_lr: inner adaptation learning rate
            inner_steps: number of PPO steps per inner adaptation
            meta_batch_size: number of tasks per meta-iteration
            ppo_kwargs: additional kwargs forwarded to PPO (must include device, lr, gamma, etc.)
        """
        self.base_ppo_cls    = base_ppo_cls
        self.task_sampler    = task_sampler
        self.device          = torch.device(device)
        self.meta_lr         = meta_lr
        self.inner_lr        = inner_lr
        self.inner_steps     = inner_steps
        self.meta_batch_size = meta_batch_size
        # ensure PPO constructions use the desired device
        ppo_kwargs["device"] = self.device
        self.ppo_kwargs      = ppo_kwargs

        # setup logging and checkpoint directory
        self.tb_writer = SummaryWriter("./tb_logs_meta")
        self.save_dir   = "./checkpoints_meta"
        os.makedirs(self.save_dir, exist_ok=True)

        # initialize the meta-agent
        init_env = self.task_sampler()
        self.meta_agent = self.base_ppo_cls(init_env, **self.ppo_kwargs)
        self.meta_agent.policy.to(self.device)

    def meta_train(self, meta_iterations=100):
        """Perform meta-training over multiple outer iterations."""
        # strip callback from inner adaptations
        inner_kwargs = {k: v for k, v in self.ppo_kwargs.items() if k != 'callback'}

        for itr in range(1, meta_iterations + 1):
            # snapshot current meta policy parameters
            meta_params = parameters_to_vector(
                self.meta_agent.policy.parameters()
            ).detach()

            adapted_params = []
            rewards = []
            actor_losses = []
            critic_losses = []

            for _ in range(self.meta_batch_size):
                # sample new task and clone
                env_clone = self.task_sampler()
                clone = self.base_ppo_cls(env_clone, **inner_kwargs)
                # load meta weights
                clone.policy.load_state_dict(
                    self.meta_agent.policy.state_dict()
                )
                # move clone to CPU for adaptation
                cpu = torch.device('cpu')
                clone.policy.to(cpu)
                clone.device = cpu
                # set inner-loop lr
                for pg in clone.actor_optimizer.param_groups:
                    pg['lr'] = self.inner_lr
                for pg in clone.critic_optimizer.param_groups:
                    pg['lr'] = self.inner_lr
                # run adaptation
                clone.learn(steps=self.inner_steps * clone.steps_batch)

                # record adapted parameters
                adapted_params.append(
                    parameters_to_vector(clone.policy.parameters())
                )
                # record stats
                if hasattr(clone, 'reward_history'):
                    rewards.append(clone.reward_history[-1])
                if hasattr(clone, 'actor_loss_history'):
                    actor_losses.append(clone.actor_loss_history[-1])
                if hasattr(clone, 'critic_loss_history'):
                    critic_losses.append(clone.critic_loss_history[-1])
                # cleanup
                if hasattr(env_clone, 'close'):
                    env_clone.close()

            # compute means & shift
            mean_adapt = torch.stack(adapted_params).mean(dim=0)
            # move mean_adapt to GPU to match meta_params
            mean_adapt = mean_adapt.to(meta_params.device)
            shift_norm = (mean_adapt - meta_params).norm().item()

            # logging stats
            avg_reward = float(np.mean(rewards)) if rewards else 0.0
            avg_actor  = float(np.mean(actor_losses)) if actor_losses else 0.0
            avg_critic = float(np.mean(critic_losses)) if critic_losses else 0.0

            self.tb_writer.add_scalar('Meta/Iteration',           itr,    itr)
            self.tb_writer.add_scalar('Meta/AvgAdaptedReward',    avg_reward, itr)
            self.tb_writer.add_scalar('Meta/AvgAdaptedActorLoss', avg_actor,  itr)
            self.tb_writer.add_scalar('Meta/AvgAdaptedCriticLoss',avg_critic, itr)
            self.tb_writer.add_scalar('Meta/ParamShiftNorm',      shift_norm, itr)
            self.tb_writer.flush()

            # checkpoint
            if itr % 12500 == 0:
                ckpt = os.path.join(self.save_dir, f"meta_agent_{itr}")
                self.meta_agent.save(ckpt)
                print(f"[Checkpoint] Saved meta_agent to {ckpt}")

            # perform first-order meta update
            with torch.no_grad():
                new_meta = meta_params + self.meta_lr * (mean_adapt - meta_params)
                vector_to_parameters(new_meta, self.meta_agent.policy.parameters())

            print(f"Meta iteration {itr}/{meta_iterations} complete")

    def evaluate(self, eval_env, episodes=5):
        """Evaluate the current meta-agent on fresh tasks."""
        # temporarily move policy to CPU for consistency with numpy observations
        cpu = torch.device('cpu')
        orig_device = self.device
        self.meta_agent.policy.to(cpu)
        self.meta_agent.device = cpu

        total_return = 0.0
        for _ in range(episodes):
            obs_tuple = eval_env.reset()
            obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            done = False
            ep_r = 0.0
            while not done:
                action, _ = self.meta_agent.predict(obs)
                step_ret = eval_env.step(action)
                if len(step_ret) == 5:
                    obs, reward, term, trunc, _ = step_ret
                    done = bool((term | trunc).any())
                else:
                    obs, reward, done, _ = step_ret
                    done = bool(np.asarray(done).any())
                # accumulate scalar return
                reward_arr = np.array(reward)
                ep_r += float(reward_arr.sum())  # Sum rewards across all envs
            total_return += ep_r

        # restore to original device
        self.meta_agent.policy.to(orig_device)
        self.meta_agent.device = orig_device

        return total_return / episodes