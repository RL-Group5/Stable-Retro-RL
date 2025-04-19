import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.callbacks import BaseCallback
import retro
from retro import Actions
from ppo import PPO
from meta import Meta 
import os

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


def make_retro(*, game, state=None, scenario=None, players=1, max_episode_steps=None):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, scenario=scenario, use_restricted_actions=Actions.DISCRETE)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    return env


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
            path = f"{self.save_path}/model_{self.num_timesteps}"
            self.model.save(path)
            if self.verbose:
                print(f"Saved model to {path}", flush=True)
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

    # Vectorized env parameters
    def build_vec_env(n_envs):
        return SubprocVecEnv([make_env_fn(args.game, args.state, args.scenario)] * n_envs)

    if args.meta:
        # --- Meta-training ---
        def task_sampler():
            # sample 4 parallel envs each task
            return SubprocVecEnv([make_env_fn(args.game, args.state, args.scenario)] * 4)

        meta = Meta(
            PPO, task_sampler,
            meta_lr=1e-3, inner_lr=1e-4,
            inner_steps=1, meta_batch_size=4,
            callback=SaveOnStepCallback(50000, "./checkpoints", verbose=1),
            # PPO hyperparameters
            steps_batch=256, steps_episode=256,
            updates_per_iteration=4,
            lr=2.5e-4, gamma=0.99
        )
        meta.meta_train(meta_iterations=20)

        # After meta-training, play with meta_agent
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
        # --- Play using a saved PPO model ---
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
        # --- Vanilla PPO training ---
        venv = build_vec_env(8)
        venv = VecFrameStack(venv, n_stack=4)
        venv = VecTransposeImage(venv)
        callback = SaveOnStepCallback(save_freq=50000, save_path="./checkpoints", verbose=1)

        model = PPO(
            venv,
            callback=callback,
            steps_batch=256,
            steps_episode=256,
            updates_per_iteration=4,
            lr=2.5e-4,
            gamma=0.99,
        )
        model.learn(steps=500)
        model.save("./checkpoints/ppo_model_final")
        print("Done training.")


if __name__ == "__main__":
    main()
