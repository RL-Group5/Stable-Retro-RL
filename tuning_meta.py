import argparse
import os
import optuna
import retro
from retro import Actions
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from ppo import PPO
from meta import Meta

# --- re-import the env builder used by retros.py ---
import gymnasium as gym
from retros import make_retro, wrap_deepmind_retro


def make_env_fn(game, state, scenario=None):
    def _init():
        env = make_retro(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env
    return _init

# tuning settings
GAME = "MortalKombatII-Genesis"
STATE = "LiuKangVsShaoKahn_VeryHard_15"
N_ENVS = 3
TRAIN_META_ITERS = 15
EVAL_EPISODES    = 3


def objective(trial):
    # sample meta‐learning hyperparams
    meta_lr        = trial.suggest_float("meta_lr",       1e-5, 1e-2, log=True)
    inner_lr       = trial.suggest_float("inner_lr",      1e-6, 1e-3, log=True)
    inner_steps    = trial.suggest_int("inner_steps",     1,    3)
    meta_batch_sz  = trial.suggest_int("meta_batch_size", 2,    4)

    # build task sampler
    def task_sampler():
        venv = SubprocVecEnv([make_env_fn(GAME, STATE)] * N_ENVS)
        venv = VecFrameStack(venv, n_stack=4)
        venv = VecTransposeImage(venv)
        return venv

    # instantiate Meta with GPU support via SB3 device arg
    meta = Meta(
        base_ppo_cls=PPO,
        task_sampler=task_sampler,
        device="cuda",                # use GPU for all neural-net computations
        meta_lr=meta_lr,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        meta_batch_size=meta_batch_sz,
        # forward default PPO settings
        lr=2.5e-4,
        gamma=0.99,
        steps_batch=256,
        updates_per_iteration=2,
        callback=None
    )

    # meta-train
    meta.meta_train(meta_iterations=TRAIN_META_ITERS)

    # evaluate on fresh tasks
    eval_env = task_sampler()
    avg_ret = meta.evaluate(eval_env, episodes=EVAL_EPISODES)
    return avg_ret


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=30)

    print("Best trial:\n", study.best_trial.params)
