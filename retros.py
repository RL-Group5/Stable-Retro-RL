
import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from ppo import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

import retro
from retro import Actions

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
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
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state,use_restricted_actions=Actions.DISCRETE, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

from stable_baselines3.common.callbacks import BaseCallback
import os

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="MortalKombatII-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--model-path", type=str, default=None, help="Path to a saved PPO model")
    parser.add_argument("--play", action="store_true", help="Play the game using a trained model")
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario, players=1)
        env = wrap_deepmind_retro(env)
        return env

    if args.play and args.model_path:
        env = DummyVecEnv([make_env])
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        model = PPO(env)
        model.load_model(args.model_path)

        obs = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done.any():
                obs = env.reset()

    else:
        venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env]*8), n_stack=4))
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
        model.learn(500)
        model.save("./checkpoints/ppo_model_final")  
        print("Done training.")



if __name__ == "__main__":
    main()