import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.tensorboard import SummaryWriter
import os

class Meta:
    def __init__(self, base_ppo_cls, task_sampler,
                 meta_lr=1e-3, inner_lr=1e-4,
                 inner_steps=1, meta_batch_size=5,
                 **ppo_kwargs):
        """
        Args:
            base_ppo_cls: class of PPO agent
            task_sampler: fn returning a fresh vec-env instance
            meta_lr: outer meta-update learning rate
            inner_lr: learning rate for per-task adaptation
            inner_steps: iterations of PPO adapt per task
            meta_batch_size: number of tasks per meta-iteration
            ppo_kwargs: kwargs forwarded to PPO constructor
        """
        self.base_ppo_cls = base_ppo_cls
        self.task_sampler = task_sampler
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.ppo_kwargs = ppo_kwargs
        self.tb_writer = SummaryWriter("./tb_logs_meta")
        self.save_dir = "./checkpoints_meta"
        os.makedirs(self.save_dir, exist_ok=True)

        # initialize your meta_agent as before
        init_env = self.task_sampler()
        self.meta_agent = base_ppo_cls(init_env, **ppo_kwargs)

    def meta_train(self, meta_iterations=100):
        for itr in range(1, meta_iterations + 1):
            # snapshot current meta parameters
            meta_params = parameters_to_vector(self.meta_agent.policy.parameters())
            adapted_params = []

            self.tb_writer.add_scalar('Meta/Iteration', itr, itr)
            self.tb_writer.add_scalar('Meta/LR', self.meta_lr, itr)
            self.tb_writer.flush()
            rewards, actor_losses, critic_losses = [], [], []

            for _ in range(self.meta_batch_size):
                # sample new task env
                task_env = self.task_sampler()
                # fresh PPO clone
                clone = self.base_ppo_cls(task_env, **self.ppo_kwargs)
                # copy meta weights
                clone.policy.load_state_dict(self.meta_agent.policy.state_dict())
                # set inner learning rates
                clone.actor_optimizer.param_groups[0]['lr'] = self.inner_lr
                clone.critic_optimizer.param_groups[0]['lr'] = self.inner_lr
                # perform adaptation
                clone.learn(steps=self.inner_steps * clone.steps_batch)
                # collect adapted weights
                adapted_params.append(parameters_to_vector(clone.policy.parameters()))
                if hasattr(clone, "reward_history") and clone.reward_history:
                    rewards.append(clone.reward_history[-1])
                if hasattr(clone, "actor_loss_history") and clone.actor_loss_history:
                    actor_losses.append(clone.actor_loss_history[-1])
                if hasattr(clone, "critic_loss_history") and clone.critic_loss_history:
                    critic_losses.append(clone.critic_loss_history[-1])
                # cleanup
                if hasattr(task_env, 'close'):
                    task_env.close()

            # — compute means for logging —
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0           
            avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0.0
            avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0
            # compute mean of adapted params
            mean_adapted = torch.stack(adapted_params).mean(dim=0)
           # measure how far the meta‑params moved
            shift_norm = (mean_adapted - meta_params).norm().item()

            # — TensorBoard logging —
            self.tb_writer.add_scalar('Meta/Iteration',        itr, itr)
            self.tb_writer.add_scalar('Meta/LR',               self.meta_lr, itr)
            self.tb_writer.add_scalar('Meta/AvgAdaptedReward', avg_reward, itr)
            self.tb_writer.add_scalar('Meta/AvgAdaptedActorLoss',  avg_actor_loss, itr)
            self.tb_writer.add_scalar('Meta/AvgAdaptedCriticLoss', avg_critic_loss, itr)
            self.tb_writer.add_scalar('Meta/ParamShiftNorm',      shift_norm, itr)
            self.tb_writer.flush()

            if itr % 12500 == 0:
                ckpt = os.path.join(self.save_dir, f"meta_agent_{itr}")
                self.meta_agent.save(ckpt)
                print(f"[Checkpoint] Saved meta_agent to {ckpt}")

            # meta-update
            new_meta = meta_params + self.meta_lr * (mean_adapted - meta_params)
            vector_to_parameters(new_meta, self.meta_agent.policy.parameters())

            print(f"Meta iteration {itr}/{meta_iterations} complete")

    def evaluate(self, eval_env, episodes=5):
        total = 0
        for _ in range(episodes):
            obs, done = eval_env.reset(), False
            ep_r = 0
            while not done:
                act, _ = self.meta_agent.predict(obs)
                obs, r, done, _ = eval_env.step(act)
                ep_r += r
            total += ep_r
        return total / episodes
