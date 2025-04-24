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


## Module Documentation

### `main.py`
**Purpose:** Starting point for training and evaluating agents. Parses command-line arguments to determine the operation mode (to train or evaluate) and initializes the corresponding processes.

**Key Functions:**
- `main()`: Loads arguments, initializes environment, model, and calls either `train_agent` or `evaluate_agent` depending on mode.
- `parse_arguments()`: Defines and processes command-line options like `--env`, `--mode`, and `--total-timesteps`.

---

### `train.py`
**Purpose:** Orchestrates training: it drives environment-agent interaction, data collection, and model updates.

**Key Function:**
- `train_agent(env, agent, config)`: For a given number of timesteps, collects rollouts and calls the agent's `update()` method. Also handles logging and saving.


---
### `tuning_ppo.py`

**Purpose:**  This script performs automated hyperparameter tuning for the PPO agent using Optuna. The goal is to optimize PPO-specific parameters like learning rates, clipping epsilon, and GAE parameters within the Mortal Kombat II environment.

**Detailed Breakdown:**
-`Libraries:` Uses optuna, retro, and Stable-Baselines3 wrappers to handle environment and PPO logic.
-`make_env()` function: Initializes the Retro environment with a specific game state, applies StochasticFrameSkip, and wraps it with DeepMind-style preprocessing (WarpFrame). Returns an environment constructor.
-`objective(trial)`: Samples hyperparameters such as:
    -`lr`: learning rate for actor
    -`clr`: learning rate for critic
    -`gamma, gae_lambda`: discount and GAE parameters
    -`steps_batch, updates_per_iteration, frame_skip, stickprob`: sampling and dynamics tuning
-Trains the PPO agent using these parameters for 50,000 timesteps
-Uses the latest reward_history[-1] as Optuna's optimization target

**Notes:**
-PPO implementation is custom (not Stable-Baselines3 PPO)
-Tightly coupled with ppo.py implementation logic

---
### `tuning_hppo.py`

**Purpose:** Performs hyperparameter optimization for Hierarchical PPO (HPPO), tuning both manager and worker policies.

**Detailed Breakdown:**
-Similar structure to tuning_ppo.py but uses HPPO instead

-Additional hyperparameters:
    -manager_lr, worker_lr: separate learning rates
    -option_duration: controls how frequently the manager switches options
    -clip_ratio, steps_per_epoch, train_epochs
-Vectorized environment (SubprocVecEnv) and frame stackers are used
-Returns the final reward as optimization score

**Notes:**
Designed to explore temporal abstraction through options
Heavily references hppo.py for internal model behavior

---

### `tuning_meta.py`

**Purpose:** Tune a Meta-Learning version of PPO (not ultimately used in final evaluation).

**Detailed Breakdown:**
-Similar structure to tuning_ppo.py, but references a meta-wrapper over PPO
-Uses Optuna to optimize both inner PPO hyperparameters and meta-specific learning rates
-Includes logic for running multiple sampled environments or seeds (implied by task_sampler in meta.py)

**Notes:**
Useful for research purposes or generalization studies
Not part of final Easy Mode PPO/HPPO setup

---

### `train_hppo.py`
**Purpose:** CLI entry script for training the HPPO agent with logging and saving functionality.

**Detailed Breakdown:**
-CLI parser to set arguments for:
    -Game, State, Scenario
    -Total steps, Number of environments, Option duration
-Logging:
    -Uses TensorBoard (SummaryWriter) for recording rewards, losses
    -SaveOnStepCallback: Custom class to save model periodically during training
-Constructs SubprocVecEnv with Retro environments
-Loads HPPO class, initiates training via .learn()
-Saves final model to checkpoints_hppo/MortalKombatII-Genesis_final.pt

---

### `retros.py`
**Purpose:** Defines custom wrappers to modify frame skip behavior for retro games.

**Detailed Breakdown:**
-StochasticFrameSkip class:
    -Takes n steps per action
    -Randomly repeats previous action with probability stickprob
    -Useful for simulating sticky keys and improving robustness
    
**Notes:**
Important for PPO and HPPO training diversity
Used across training and tuning scripts


---


### `ppo.py`
**Class:** `PPOAgent`
**Purpose:** Implements Proximal Policy Optimization with a shared rollout buffer.

**Architecture:**
- Actor: MLP with softmax output for discrete action spaces.
- Critic: MLP with a single scalar value output.

**Detailed Breakdown:**
CNNPolicy:
    -Convolutional layers followed by separate actor and critic heads
    -Actor outputs logits, critic outputs value estimation
PPO class:
-Implements policy learning with clipped surrogate objective
-`episodes()`: Collects rollout data (states, actions, returns)
-`learn()`: Performs mini-batch updates using advantage estimates
-`evaluate()`: Computes values and log-probs for batch updates
-`plot_results()`: Visualization of training curves

**Notes:**
Self-contained, not reliant on SB3
Handles reward normalization and logging internally



---

### `hppo.py`
**Class:** `HPPOAgent`
**Purpose:** Implements a two-tier PPO structure. Manager and Worker operate at different levels of abstraction.


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
- Outperformed PPO on longer horizons in sparse reward settings
- Demonstrated strong no-damage win performance in Mortal Kombat II (easy mode)


---
## `meta.py`

**Purpose:**  Meta-learning wrapper to adapt PPO across multiple tasks.

**Detailed Breakdown:**
Meta class:
Wraps PPO model
Uses gradient-based updates to modify PPO's inner-loop optimizer
Contains outer-loop (meta) learning rate
Intended for tasks with non-stationary dynamics
Notes:
Not used in the final Mortal Kombat experiments

---

## `setup.py`

**Purpose:** CMake-based setup script to compile the native Retro emulator backend.

**Detailed Breakdown:**
CMakeBuild: custom setuptools extension
-Detects Python paths, calls cmake, builds C++ files
-Installs native .dylib or .so libraries required for Retro gameplay rendering and logic

**Notes:**
-Required only if building stable-retro from source
-Can be skipped if using Homebrew installation

Cloning the stable-retro GitHub repository retrieves the full source code, including the C++ emulator backends, configuration files for supported game platforms (e.g., Genesis, SNES), and Python bindings. This gives users complete control over building and modifying the library from source, rather than relying on prebuilt binaries.

**setup.py Functionality:**

The setup.py file is a custom Python installer script that integrates with CMake to:
-Detect the active Python environment
-Build native emulator cores (like libretro) using C++ toolchains
-Link the compiled binaries to Python via a custom extension module
-Register the package so it can be imported with import retro

This setup is required when using the local source version of stable-retro, especially when extending it (e.g., adding new environments or recompiling for compatibility with a specific system).

