import argparse
import os
import gymnasium as gym
import retro
from retro import Actions
import torch
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from hppo import HPPO, ACTION_GROUPS


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        super().__init__(env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            if self.curac is None:
                self.curac = ac
            elif i == 0 and self.rng.rand() > self.stickprob:
                self.curac = ac
            elif i == 1:
                self.curac = ac

            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)

            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, scenario=None):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, scenario=scenario, use_restricted_actions=Actions.DISCRETE)
    return StochasticFrameSkip(env, n=4, stickprob=0.25)


def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def make_env_fn(game, state, scenario):
    def _init():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env
    return _init


class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"model_{self.num_timesteps}.pt")
            self.model.save(path)
            if self.verbose:
                print(f"Saved model to {path}", flush=True)
        return True


class TensorboardCallback(BaseCallback):
    def __init__(self, writer: SummaryWriter, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        ts = self.num_timesteps
        if ts % self.log_freq == 0:
            # Episodic reward from Monitor
            if 'episode' in self.locals['infos'][0]:
                ep_rew = self.locals['infos'][0]['episode']['r']
                self.writer.add_scalar('Reward/episode', ep_rew, ts)

            # Custom logs if model exposes them
            if hasattr(self.model, 'reward_history') and self.model.reward_history:
                self.writer.add_scalar('Reward/average', self.model.reward_history[-1], ts)
            if hasattr(self.model, 'manager_loss_history') and self.model.manager_loss_history:
                self.writer.add_scalar('Loss/manager', self.model.manager_loss_history[-1], ts)
            if hasattr(self.model, 'worker_loss_history') and self.model.worker_loss_history:
                self.writer.add_scalar('Loss/worker', self.model.worker_loss_history[-1], ts)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--option-duration", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=1024)
    parser.add_argument("--game", default="MortalKombatII-Genesis")
    parser.add_argument("--state", default="Level1.JaxVsBaraka")
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    # TensorBoard setup
    tb_dir = os.path.join("./tb_logs", args.game)
    writer = SummaryWriter(tb_dir)
    #tb_callback = TensorboardCallback(writer, log_freq=10)
    save_cb = SaveOnStepCallback(save_freq=50000, save_path="./checkpoints_hppo", verbose=1)

    # Log hyperparameters
    for k, v in vars(args).items():
        writer.add_text(f"Hyperparam/{k}", str(v), 0)

    # Create vectorized environment with monitor
    env_fns = [make_env_fn(args.game, args.state, args.scenario) for _ in range(args.n_envs)]
    venv = SubprocVecEnv(env_fns)
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)

    print(f"Raw observation shape: {venv.observation_space.shape}")
    print(f"Action dimension: {venv.action_space.n}")
    print(f"Number of options: {len(ACTION_GROUPS)}")
    print(f"Starting training with {args.n_envs} environments")
    print(f"Total steps: {args.total_steps}, Steps per epoch: {args.steps_per_epoch}")
    print(f"Option duration: {args.option_duration} steps")

    agent = HPPO(
        venv,
        device="cuda" if torch.cuda.is_available() else "cpu",
        option_duration=args.option_duration,
        steps_per_epoch=args.steps_per_epoch,
        writer=writer
    )

    try:
        agent.learn(args.total_steps)
        agent.save("./checkpoints_hppo/MortalKombatII-Genesis_final.pt")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save("./checkpoints_hppo/MortalKombatII-Genesis_final.pt")
    finally:
        writer.close()
        venv.close()


if __name__ == "__main__":
    main()
