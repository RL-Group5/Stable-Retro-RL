import argparse
import os
import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.callbacks import BaseCallback
import retro
from retro import Actions
from ppo import PPO
from meta import Meta


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


def make_retro(*, game, state=None, scenario=None, players=1):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, scenario=scenario, use_restricted_actions=Actions.DISCRETE)
    return StochasticFrameSkip(env, n=4, stickprob=0.25)


def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"model_{self.num_timesteps}")
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
        if ts % self.log_freq == 0 and hasattr(self.model, 'reward_history'):
            # Log average reward
            avg_r = self.model.reward_history[-1] if self.model.reward_history else 0
            self.writer.add_scalar('Reward/average', avg_r, ts)
            # Log losses if available
            if hasattr(self.model, 'actor_loss_history') and self.model.actor_loss_history:
                self.writer.add_scalar('Loss/actor', self.model.actor_loss_history[-1], ts)
            if hasattr(self.model, 'critic_loss_history') and self.model.critic_loss_history:
                self.writer.add_scalar('Loss/critic', self.model.critic_loss_history[-1], ts)
        return True


class CombinedCallback(BaseCallback):
    def __init__(self, callbacks):
        super().__init__(0)
        self.callbacks = callbacks

    def _on_step(self) -> bool:
        for cb in self.callbacks:
            cb.model = self.model
            cb.num_timesteps = self.num_timesteps
            cb._on_step()
        return True


def make_env_fn(game, state, scenario):
    def _init():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="MortalKombatII-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--meta", action="store_true", help="Run meta-training instead of vanilla PPO")
    args = parser.parse_args()

    # TensorBoard writer
    tb_dir = os.path.join("./tb_logs", args.game)
    writer = SummaryWriter(tb_dir)
    tb_callback = TensorboardCallback(writer, log_freq=1000)

    # Save callback
    save_cb = SaveOnStepCallback(save_freq=50000, save_path="./checkpoints", verbose=1)
    combined_cb = CombinedCallback([tb_callback, save_cb])

    # Build vectorized env function
    def build_vec_env(n_envs):
        return SubprocVecEnv([make_env_fn(args.game, args.state, args.scenario)] * n_envs)

    if args.meta:
        # Meta-training: wrap each task env with frame stacking and transpose
        def task_sampler():
            venv = SubprocVecEnv([make_env_fn(args.game, args.state, args.scenario)] * 5)
            venv = VecFrameStack(venv, n_stack=4)
            venv = VecTransposeImage(venv)
            return venv

        meta = Meta(
            PPO, task_sampler,
            meta_lr=1e-3, inner_lr=1e-4,
            inner_steps=1, meta_batch_size=4,
            callback=combined_cb,
            steps_batch=512, steps_episode=256,
            updates_per_iteration=4,
            lr=2.5e-4, gamma=0.99
        )
        # Meta-train
        meta.meta_train(meta_iterations=250001)

        # Run meta-agent with correct obs shape preprocessing
        play_env = DummyVecEnv([make_env_fn(args.game, args.state, args.scenario)])
        play_env = VecFrameStack(play_env, n_stack=4)
        play_env = VecTransposeImage(play_env)
        obs = play_env.reset()
        while True:
            action, _ = meta.meta_agent.predict(obs)
            obs, reward, done, _ = play_env.step(action)
            play_env.render()
            if done.any():
                obs = play_env.reset()

    elif args.play and args.model_path:
        # Play with saved PPO
        play_env = DummyVecEnv([make_env_fn(args.game, args.state, args.scenario)])
        play_env = VecFrameStack(play_env, n_stack=4)
        play_env = VecTransposeImage(play_env)
        model = PPO(play_env)
        model.load_model(args.model_path)

        obs = play_env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, _ = play_env.step(action)
            play_env.render()
            if done.any():
                obs = play_env.reset()

    else:
        # Vanilla PPO training
        venv = build_vec_env(3)
        venv = VecFrameStack(venv, n_stack=4)
        venv = VecTransposeImage(venv)

        model = PPO(
            venv,
            callback=combined_cb,
            steps_batch=824,
            steps_episode=824,
            updates_per_iteration=4,
            lr=2.5e-4,
            gamma=0.99,
        )
        model.learn(steps=600000)
        model.save("./checkpoints/ppo_model_final")
        print("Done training.")


if __name__ == "__main__":
    main()
