# Stable-Retro-RL

## Project Overview

**Repository:** [RL-Group5/Stable-Retro-RL](https://github.com/RL-Group5/Stable-Retro-RL)  
**Team Members:** Khai, Aman, Zane, Zach, Rahul  
**Purpose:** Develop and evaluate reinforcement learning agents using the Rainbow DQN algorithm within the Mortal Kombat II environment, utilizing the Stable-Retro framework.

---

## Directory Structure

README.md
requirements.txt
main.py
train.py
evaluate.py
utils/
├── env_wrapper.py
├── replay_buffer.py
├── models/
│   ├── dqn.py
│   ├── meta_dqn.py
│   ├── ppo.py
│   └── hppo.py
└── helpers/
├── logger.py
└── plotter.py


---

## Getting Started & Environment Setup

### Prerequisites

- **OS:** Ubuntu 20.04 or equivalent Linux with SDL2/OpenGL  
- **Python:** 3.9 or 3.10  
- **Tools:** git, pip, (optional: conda)

### Installation

```bash
git clone https://github.com/RL-Group5/Stable-Retro-RL.git
cd Stable-Retro-RL
conda create -n stk_rl python=3.9  # optional
conda activate stk_rl
pip install -r requirements.txt
```

###Pinned Dependency Versions

stable-baselines3==1.8.0  
gym-retro==0.8.0  
torch==1.12.1  
numpy==1.23.1  
opencv-python==4.5.5.64  
optuna==3.0.0  
tensorboard==2.12.1  

###Docker (alternative)

FROM ubuntu:22.04  
RUN apt-get update && \  
    apt-get install -y python3-pip libsdl2-dev libgl1-mesa-dev  
WORKDIR /app  
COPY . /app  
RUN pip3 install -r requirements.txt  
CMD ["python3", "main.py", "--mode", "train", "--env", "SuperTuxKart-Ubuntu"]  

docker build -t stk-rl .  
docker run --rm stk-rl  

### Quickstart

#### Training
python main.py --mode train --env MortalKombatII-Genesis --seed 42 --total-timesteps 100000

#### Evaluation
python main.py --mode evaluate --env Mortal KombatII-Genesis --load-path logs/checkpoints/latest.zip --episodes 20

## Module Documentation



### main.py
#### Purpose: Starting point for training and evaluating agents. Parses command-line arguments to determine the operation mode (either to train or evaluate) and initializes the corresponding processes.

#### Functions:

parse_arguments() - Parses command-line arguments to configure training or evaluation parameters. --env selects which Retro environment to load. . --seed is also provided so users can reproduce runs precisely Returns namespace object containing configuration settings.



main() - Main function that builds the environment (via EnvWrapper), instantiates the appropriate agent class from utils/models/*, then dispatches to either train.train_agent or evaluate.evaluate_agent based on a --mode flag.

### train.py
#### Purpose: Handles agent-environment interaction loop and model updates.
#### Function:
train_agent(env, agent, config) - Trains the agent within the provided environment using configurations specified in config
##### Parameters:


env: The environment instance.


agent: The reinforcement learning agent.


config: Configuration settings for training.




### evaluate.py
#### Purpose: Runs agent in evaluation mode to gather metrics.
#### Functions:
evaluate_agent(env, agent, episodes) - Runs the agent in the environment for a specified number of episodes and records performance metrics

##### Parameters:


env: The environment instance.


agent: The trained agent to evaluate.


episodes: Number of episodes to run for evaluation.



### utils/env_wrapper.py
#### Purpose: Provides a wrapper for the environment to ensure compatibility with the agent's expected input and output formats.
#### Class:
EnvWrapper with reset(), step(), render() - Wraps the original environment to preprocess observations and actions, and to handle any environment-specific quirks.


##### Methods:


reset(): Resets the environment and returns the initial observation.

step(action): Applies the given action to the environment and returns the resulting observation, reward, done flag, and info dictionary.

render(): Renders the current state of the environment.



### utils/replay_buffer.py
#### Purpose: Stores and samples transitions for off-policy training.
#### Class:
ReplayBuffer with add(), sample(), __len__() - Stores experiences (state, action, reward, next state, done) and allows for random sampling to break correlation between sequential data.
##### Methods:

add(state, action, reward, next_state, done): Appends a new experience to the buffer. Computes and stores initial priority if priority replay is enabled

sample(batch_size): Samples a batch of experiences from the buffer. Returns a tuple of batches (states, actions, rewards, next_states, dones, weights, indices). Returns importance-sampling weights when using prioritized replay

__len__(): Returns the current size of the buffer. Used to check buffer warm-up status



### utils/models/dqn.py
#### Purpose: Defines the Deep Q-Network (DQN) agent architecture and training methods.
#### Class:
DQNAgent with select_action(), train_step(), update_target_network() - Implements the DQN algorithm with support for extensions like Double DQN and Dueling Networks.


##### Methods:


select_action(state): Selects an action based on the current policy. (ε-greedy or NoisyNet)


train_step(batch): Performs a single training step using a batch of experiences.


update_target_network(): Updates the target network weights.



### utils/models/meta_dqn.py
#### Purpose: Implements a Meta-DQN agent that dynamically adapts its learning strategy based on environmental feedback.
#### Class: MetaDQNAgent
##### Key Methods:


select_action(state): Selects action using evolving exploration/exploitation strategy.


adapt_hyperparams(metrics): Updates learning rate or ε-schedule based on performance.


train_step(batch): Trains agent using DQN logic with meta-adaptive control.


##### Notes: Still experimental; designed for self-tuning behavior in non-stationary or complex reward settings.


### utils/models/ppo.py

#### Purpose: Implements Proximal Policy Optimization (PPO) for policy gradient training in discrete environments.
#### Class: PPOAgent
##### Key Methods:


select_action(state): Samples action from policy.


evaluate_actions(states, actions): Computes log probs, entropy, and value estimates.


compute_returns(): Uses GAE for advantage estimation.


update(rollouts): Applies clipped surrogate objective and updates actor–critic networks.


##### Notes: Uses separate actor–critic networks, GAE, entropy regularization, and multi-epoch updates for stable training.


### utils/models/hppo.py
#### Purpose: Hierarchical PPO - manager issues subgoals, worker executes primitives.
#### Class & Key Methods:
__init__() - Accepts manager/worker networks and config


select_low_level_action(state, goal) - Chooses primitive action


train() - Updates both policy levels with advantage estimates


##### Notes: Achieves perfect score on easy mode in testing. Manages high- and low-level learning cycles.


### utils/helpers/logger.py
#### Purpose: Logs scalars, histograms, hyperparameters using TensorBoard.
#### Class: Logger
##### Key Methods:
__init__(log_dir): Initializes the logging directory and backend (e.g., TensorBoard).
log_scalar(tag, value, step): Records a scalar value such as loss or reward.
log_params(params): Saves a dictionary of hyperparameters.
log_histogram(tag, values, step): Records distributions (weights, activations, etc.).
close(): Finalizes and flushes logs to disk.

##### Notes: Used across training/evaluation to visualize model behavior and learning trends.


### utils/helpers/plotter.py
#### Purpose: Generates visualizations for training diagnostics, including learning curves and parameter distributions.
#### Class: Plotter
##### Key Methods:
plot_rewards(log_dir, save_path=None, smooth_window=5): Plots smoothed episodic reward curves.
plot_loss(log_dir, save_path=None, smooth_window=5): Plots training loss over time.
plot_histogram(log_dir, tag, step, save_path=None): Visualizes value distributions at a specific training step.
##### Notes: Supports both real-time tracking and offline evaluation through exported metrics.




