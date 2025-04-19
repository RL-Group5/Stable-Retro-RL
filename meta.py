import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy

class Meta:
    """

    This wrapper will:
    1. Sample tasks (envs with varied speeds/obstacles) via `task_sampler()`.
    2. Clone the PPO agent and perform a small number of inner-loop updates on each task.
    3. Average the adapted parameters and move the meta-parameters toward that average.
    4. Produce a meta-trained policy ready to adapt quickly to new task variations.
    """
    def __init__(self, base_ppo_cls, task_sampler,
                 meta_lr=1e-3, inner_lr=1e-4,
                 inner_steps=1, meta_batch_size=5,
                 **ppo_kwargs):
        """
        Args:
            base_ppo_cls:       PPO class
            task_sampler:       Function returning a fresh env instance each call.
            meta_lr:            Outer meta-learning rate.
            inner_lr:           Learning rate for adaptation on each task.
            inner_steps:        Number of PPO update steps per task.
            meta_batch_size:    Number of tasks per meta-iteration.
            ppo_kwargs:         Keyword args forwarded to inner PPO instances.
        """
        self.task_sampler = task_sampler
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size

        # Initialize meta-agent on first sampled env
        initial_env = self.task_sampler()
        self.meta_agent = base_ppo_cls(initial_env, **ppo_kwargs)

    def meta_train(self, meta_iterations=100):
        """
        Run meta-training loop.
        For each iteration:
          - Clone and adapt the meta-agent on a batch of tasks.
          - Average their parameters and step the meta-parameters toward that average.
        """
        for itr in range(1, meta_iterations + 1):
            # Snapshot current meta-parameters
            meta_params = parameters_to_vector(self.meta_agent.policy.parameters())
            adapted_params = []

            # Inner loop over sampled tasks
            for _ in range(self.meta_batch_size):
                env = self.task_sampler()
                agent_clone = copy.deepcopy(self.meta_agent)
                # Override optimizers for inner adaptation
                agent_clone.actor_optimizer.param_groups[0]['lr'] = self.inner_lr
                agent_clone.critic_optimizer.param_groups[0]['lr'] = self.inner_lr
                # Adapt on this task
                agent_clone.learn(steps=self.inner_steps * agent_clone.steps_batch)
                # Collect adapted parameters
                adapted_params.append(
                    parameters_to_vector(agent_clone.policy.parameters())
                )
                # Clean up
                if hasattr(env, 'close'):
                    env.close()

            # Compute average of adaptations
            adapted_stack = torch.stack(adapted_params)
            mean_adapted = adapted_stack.mean(dim=0)

            # Reptile meta-update step
            updated = meta_params + self.meta_lr * (mean_adapted - meta_params)
            vector_to_parameters(updated, self.meta_agent.policy.parameters())

            print(f"Meta iteration {itr}/{meta_iterations} complete")

    def evaluate(self, eval_env, episodes=5):
        """
        Test the meta-trained policy on a new environment without inner adaptation.
        Returns the average episodic reward.
        """
        total_rewards = []
        for _ in range(episodes):
            obs, done = eval_env.reset(), False
            ep_reward = 0
            while not done:
                action, _ = self.meta_agent.predict(obs)
                obs, reward, done, _ = eval_env.step(action)
                ep_reward += reward
            total_rewards.append(ep_reward)
        return sum(total_rewards) / len(total_rewards)