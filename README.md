# Stable-Retro-RL

## Project Overview
**Repository:** [RL-Group5/Stable-Retro-RL](https://github.com/RL-Group5/Stable-Retro-RL)  
**Team Members:** Khai, Aman, Zane, Zach, Rahul  
**Project Focus:** Train and evaluate reinforcement learning agents in Mortal Kombat II using Proximal Policy Optimization (PPO) and Hierarchical PPO (HPPO) with the Gym-Retro framework.

---

## Getting Started & Setup

### Prerequisites
- Ubuntu 20.04 or compatible Linux OS
- Python 3.9 or 3.10
- Tools: pip, (optional) conda
- Gym-Retro and a copy of the Mortal Kombat II ROM

### Installation
```bash
git clone https://github.com/RL-Group5/Stable-Retro-RL.git
cd Stable-Retro-RL
conda create -n retro_rl python=3.9
conda activate retro_rl
pip install -r requirements.txt
```

### ROM Import
```bash
python -m retro.import /path/to/mk2-rom-folder
```

### Docker (Optional)
```Dockerfile
FROM ubuntu:22.04
RUN apt-get update && \
    apt-get install -y python3-pip libsdl2-dev libgl1-mesa-dev
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
CMD ["python3", "main.py", "--mode", "train", "--env", "MortalKombatII-Genesis"]
```

---

## PPO Implementation

**Overview:**
Proximal Policy Optimization (PPO) is an on-policy reinforcement learning algorithm that improves the stability and reliability of policy gradient methods. It does so by using a clipped surrogate objective to constrain policy updates and prevent large deviations.

**Key Components:**
- **Separate Actor–Critic Networks**
- **Generalized Advantage Estimation (GAE)**
- **Entropy Regularization**
- **Multi-Epoch Updates**

Each of these is elaborated below in the code modules.

---

## HPPO Implementation

**Overview:**
Hierarchical Proximal Policy Optimization (HPPO) extends PPO by introducing a two-tier structure for abstract goal-directed learning.

**Architecture:**
- **Manager (High-Level Policy):**
  - Selects discrete subgoals every fixed number of steps (e.g., "approach", "retreat", "block")
  - Receives state as input and outputs subgoal probabilities

- **Worker (Low-Level Policy):**
  - Takes both current state and current subgoal as input
  - Outputs primitive actions (e.g., jump, punch, move left)

**Workflow:**
1. Manager chooses a subgoal at fixed intervals (e.g., every `k` steps)
2. Worker executes actions under the selected subgoal for the interval
3. Both manager and worker collect rollouts into separate buffers
4. GAE is used to compute advantages for both manager and worker
5. PPO updates are applied independently to each policy using its buffer

**Why HPPO?**
- Captures temporal abstraction: manager focuses on strategy, worker focuses on execution
- Enables structured exploration and faster learning in environments with high-dimensional input and long-horizon dependencies
- Empirically shown to achieve high consistency in Mortal Kombat II (i.e., no-damage wins)

---

## Module Documentation

### `main.py`
**Purpose:** Starting point for training and evaluating agents. Parses command-line arguments to determine the operation mode (to train or evaluate) and initializes the corresponding processes.

**Key Functions:**
- `main()`: Loads arguments, initializes environment, model, logger, and calls either `train_agent` or `evaluate_agent` depending on mode.
- `parse_arguments()`: Defines and processes command-line options like `--env`, `--mode`, and `--total-timesteps`.

---

### `train.py`
**Purpose:** Orchestrates training: it drives environment-agent interaction, data collection, and model updates.

**Key Function:**
- `train_agent(env, agent, config)`: For a given number of timesteps, collects rollouts and calls the agent's `update()` method. Also handles logging and saving.

---

### `evaluate.py`
**Purpose:** Runs evaluation episodes using a pre-trained agent.

**Key Function:**
- `evaluate_agent(env, agent, episodes)`: Executes the agent in evaluation mode for a specified number of episodes and collects metrics like average reward and win rate.

---

### `utils/env_wrapper.py`
**Class:** `EnvWrapper`
**Purpose:** Wraps the Gym-Retro environment to provide consistent preprocessing:
- Frame resizing and grayscale conversion
- Frame skipping and stacking
- Normalizes observation space for neural network input

---

### `utils/replay_buffer.py`
**Class:** `RolloutBuffer`
**Purpose:** Temporary storage for PPO-style rollouts.
- Stores states, actions, rewards, dones, log-probs, and values.
- Computes returns and advantages using Generalized Advantage Estimation (GAE).

**Key Functions:**
- `store()`: Appends a single time-step transition to the buffer.
- `compute_returns_and_advantages(last_value)`: Computes discounted returns and GAE advantages.
- `get()`: Provides mini-batches for PPO updates.

---

### `utils/models/ppo.py`
**Class:** `PPOAgent`
**Purpose:** Implements Proximal Policy Optimization with a shared rollout buffer.

**Architecture:**
- Actor: MLP with softmax output for discrete action spaces.
- Critic: MLP with a single scalar value output.

**Key Methods:**
- `select_action(obs)`: Uses actor to sample an action and record its log-probability.
- `evaluate_actions(obs, actions)`: Computes entropy, log-probs, and value estimates.
- `compute_returns_and_advantages()`: Calls GAE logic to estimate advantages.
- `update()`: Performs PPO optimization:
  - Computes surrogate loss
  - Applies clipping to avoid destructive updates
  - Adds entropy bonus to encourage exploration

---

### `utils/models/hppo.py`
**Class:** `HPPOAgent`
**Purpose:** Implements a two-tier PPO structure. Manager and Worker operate at different levels of abstraction.

**Key Components:**
- `select_option(state)`: The manager chooses a subgoal/strategy every `option_duration` timesteps.
- `select_low_level_action(state, goal)`: The worker uses the state and current goal to select a primitive action.
- `train()`: Updates manager and worker independently:
  - Each level stores transitions in separate buffers
  - GAE is used for both levels
  - PPO update (clipping, entropy, MSE critic loss) is applied per policy

**Notable Behavior:**
- The manager outputs fewer, strategic decisions
- The worker translates these into detailed actions
- Demonstrated strong no-damage win performance in Mortal Kombat II (easy mode)

---

### `utils/helpers/logger.py`
**Class:** `Logger`
**Purpose:** Logs experiment metrics and configurations to TensorBoard.

**Key Methods:**
- `log_scalar(name, value, step)`: Logs scalar metrics like reward, loss.
- `log_params(params)`: Saves hyperparameters to the log.
- `log_histogram(tag, values, step)`: Visualizes network weights or activations.
- `close()`: Finalizes and saves all logs.

---

### `utils/helpers/plotter.py`
**Class:** `Plotter`
**Purpose:** Visualization utility to render training statistics.

**Key Methods:**
- `plot_rewards(log_dir, save_path, smooth_window)`: Plots average episodic return.
- `plot_loss(log_dir, save_path, smooth_window)`: Shows actor/critic loss across updates.
- `plot_histogram(log_dir, tag, step)`: Extracts and displays distribution statistics.

---

