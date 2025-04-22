import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import os

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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
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
        x = torch.flatten(x, 1)  # Flatten while preserving batch dim
        # Concatenate features with option embedding
        combined = torch.cat([x, option_embedding], dim=1)
        action_logits = self.action_head(combined)
        value = self.value_head(combined)
        return action_logits, value

    def select_action(self, obs: torch.Tensor, options: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select low-level actions using the worker network"""
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)  # Add batch dimension if needed
            option_embeddings = options.float().unsqueeze(1)
            action_logits, _ = self.forward(obs, option_embeddings)
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs=action_probs)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)
        return actions, log_probs

class HPPO:
    """Hierarchical PPO implementation"""
    def __init__(self, env, device="cuda", **kwargs):
        self.env = env
        self.device = torch.device(device)
        self.callback = kwargs.pop("callback", None)
        self._init_hyperparameters(kwargs)
        
        # Initialize networks
        obs_shape = env.observation_space.shape
        print(f"Observation shape: {obs_shape}")
        self.num_options = len(ACTION_GROUPS)
        self.action_dim = env.action_space.n
        print(f"Action dimension: {self.action_dim}")
        print(f"Number of options: {self.num_options}")

        self.manager = ManagerNetwork(obs_shape, self.num_options).to(self.device)
        self.worker = WorkerNetwork(obs_shape, self.action_dim).to(self.device)

        # Setup optimizers
        self.manager_optimizer = torch.optim.AdamW(
            self.manager.parameters(), 
            lr=self.manager_lr,
            eps=self.adam_eps
        )
        self.worker_optimizer = torch.optim.AdamW(
            self.worker.parameters(),
            lr=self.worker_lr,
            eps=self.adam_eps
        )

        # Initialize trackers
        self.reward_history = []
        self.manager_loss_history = []
        self.worker_loss_history = []
        self.num_timesteps = 0

    def _init_hyperparameters(self, hyperparameters: Dict):
        """Initialize hyperparameters"""
        self.manager_lr = hyperparameters.get("manager_lr", 3e-4)
        self.worker_lr = hyperparameters.get("worker_lr", 3e-4)
        self.adam_eps = hyperparameters.get("adam_eps", 1e-5)
        self.gamma = hyperparameters.get("gamma", 0.99)
        self.gae_lambda = hyperparameters.get("gae_lambda", 0.95)
        self.clip_ratio = hyperparameters.get("clip_ratio", 0.2)
        self.option_duration = hyperparameters.get("option_duration", 8)
        self.steps_per_epoch = hyperparameters.get("steps_per_epoch", 2048)
        self.train_epochs = hyperparameters.get("train_epochs", 10)

    def learn(self, total_steps: int):
        """Main training loop"""
        num_epochs = total_steps // self.steps_per_epoch
        obs = self.env.reset()
        current_options = None
        option_step_counter = 0
        episode_rewards = np.zeros(self.env.num_envs)
        
        for epoch in range(num_epochs):
            batch_obs = []
            batch_options = []
            batch_actions = []
            batch_option_probs = []
            batch_action_probs = []
            batch_rewards = []
            batch_values = []
            batch_dones = []

            # Collect experience
            for step in range(self.steps_per_epoch):
                self.num_timesteps += self.env.num_envs
                
                # Handle callbacks
                if self.callback:
                    if isinstance(self.callback, list):
                        for cb in self.callback:
                            cb.model = self
                            cb.num_timesteps = self.num_timesteps
                            cb._on_step()
                    else:
                        self.callback.model = self
                        self.callback.num_timesteps = self.num_timesteps
                        self.callback._on_step()
                
                # Handle stacked observations (N, C*stack, H, W)
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                if len(obs_tensor.shape) == 4:  # If stacked observation
                    N, C, H, W = obs_tensor.shape
                    obs_tensor = obs_tensor.view(N, -1, H, W)  # Combine stacked channels
                
                # Select or continue options
                if current_options is None or option_step_counter >= self.option_duration:
                    current_options, option_log_probs = self.manager.select_option(obs_tensor)
                    option_step_counter = 0
                
                # Select actions within current options
                actions, action_log_probs = self.worker.select_action(obs_tensor, current_options)
                
                # Step environment
                next_obs, rewards, dones, infos = self.env.step(actions.cpu().numpy())
                episode_rewards += rewards
                
                # Store transition
                batch_obs.append(obs)
                batch_options.append(current_options.cpu().numpy())
                batch_actions.append(actions.cpu().numpy())
                batch_option_probs.append(option_log_probs.cpu().numpy())
                batch_action_probs.append(action_log_probs.cpu().numpy())
                batch_rewards.append(rewards)
                batch_dones.append(dones)
                
                # Update counters and check termination
                option_step_counter += 1
                obs = next_obs
                
                # Reset environments that are done
                if any(dones):
                    for i, done in enumerate(dones):
                        if done:
                            self.reward_history.append(episode_rewards[i])
                            episode_rewards[i] = 0
                    if all(dones):
                        current_options = None
                        obs = self.env.reset()

            # Convert to tensors and reshape observations
            batch_obs = torch.FloatTensor(np.array(batch_obs)).to(self.device)
            if len(batch_obs.shape) == 5:  # If stacked observations
                T, N, C, H, W = batch_obs.shape
                batch_obs = batch_obs.view(T * N, -1, H, W)  # Combine time and batch dims, and stack channels
            
            batch_options = torch.LongTensor(np.array(batch_options)).to(self.device)
            batch_actions = torch.LongTensor(np.array(batch_actions)).to(self.device)
            batch_rewards = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
            
            # Update policies
            self.update_manager(batch_obs, batch_options, batch_rewards)
            self.update_worker(batch_obs, batch_actions, batch_options, batch_rewards)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                mean_reward = np.mean(self.reward_history[-10:]) if self.reward_history else 0
                print(f"Epoch {epoch+1}/{num_epochs} | Mean Reward: {mean_reward:.2f}")

    def update_manager(self, obs, options, rewards):
        """Update manager policy"""
        if len(obs.shape) == 5:  # If stacked observations
            T, N, C, H, W = obs.shape
            obs = obs.view(T * N, -1, H, W)  # Combine time and batch dims, and stack channels
            
        option_logits, values = self.manager(obs)
        
        # Compute advantages and returns
        advantages = self.compute_advantages(rewards, values)
        returns = advantages + values.detach().view(-1)
        
        # PPO update
        for _ in range(self.train_epochs):
            new_option_logits, new_values = self.manager(obs)
            
            # Compute ratio and clipped objective
            log_probs = F.log_softmax(new_option_logits, dim=-1)
            old_log_probs = F.log_softmax(option_logits.detach(), dim=-1)
            
            ratio = torch.exp(
                log_probs.gather(1, options.view(-1, 1)) - 
                old_log_probs.gather(1, options.view(-1, 1))
            )
            
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages.unsqueeze(1)
            loss = -torch.min(ratio * advantages.unsqueeze(1), clip_adv).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.view(-1), returns)
            
            # Total loss
            total_loss = loss + 0.5 * value_loss
            
            # Update
            self.manager_optimizer.zero_grad()
            total_loss.backward()
            self.manager_optimizer.step()
            
            self.manager_loss_history.append(total_loss.item())

    def update_worker(self, obs, actions, options, rewards):
        """Update worker policy"""
        if len(obs.shape) == 5:  # If stacked observations
            T, N, C, H, W = obs.shape
            obs = obs.view(T * N, -1, H, W)  # Combine time and batch dims, and stack channels
            
        option_embeddings = options.float().unsqueeze(1)
        action_logits, values = self.worker(obs, option_embeddings)
        
        # Compute advantages and returns
        advantages = self.compute_advantages(rewards, values)
        returns = advantages + values.detach().view(-1)
        
        # PPO update
        for _ in range(self.train_epochs):
            new_action_logits, new_values = self.worker(obs, option_embeddings)
            
            # Compute ratio and clipped objective
            log_probs = F.log_softmax(new_action_logits, dim=-1)
            old_log_probs = F.log_softmax(action_logits.detach(), dim=-1)
            
            ratio = torch.exp(
                log_probs.gather(1, actions.view(-1, 1)) - 
                old_log_probs.gather(1, actions.view(-1, 1))
            )
            
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages.unsqueeze(1)
            loss = -torch.min(ratio * advantages.unsqueeze(1), clip_adv).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.view(-1), returns)
            
            # Total loss
            total_loss = loss + 0.5 * value_loss
            
            # Update
            self.worker_optimizer.zero_grad()
            total_loss.backward()
            self.worker_optimizer.step()
            
            self.worker_loss_history.append(total_loss.item())

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute GAE advantages"""
        # Reshape rewards if needed
        if len(rewards.shape) == 1:
            rewards = rewards.view(-1, self.env.num_envs)
        
        # Reshape values if needed
        if len(values.shape) == 1:
            values = values.view(-1, self.env.num_envs)
        
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.env.num_envs).to(self.device)
        
        for t in reversed(range(rewards.size(0))):
            if t == rewards.size(0) - 1:
                next_value = torch.zeros_like(values[0])
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
        # Flatten advantages to match the batch dimension
        return advantages.view(-1)

    def save(self, path: str):
        """Save model weights"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'manager_state_dict': self.manager.state_dict(),
            'worker_state_dict': self.worker.state_dict(),
            'manager_optimizer': self.manager_optimizer.state_dict(),
            'worker_optimizer': self.worker_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}", flush=True)

    def load(self, path: str):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.manager.load_state_dict(checkpoint['manager_state_dict'])
        self.worker.load_state_dict(checkpoint['worker_state_dict'])
        self.manager_optimizer.load_state_dict(checkpoint['manager_optimizer'])
        self.worker_optimizer.load_state_dict(checkpoint['worker_optimizer'])
        print(f"Model loaded from {path}") 