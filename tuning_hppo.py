import optuna
import retro
import numpy as np
from retro import Actions
from retros import StochasticFrameSkip
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from hppo import HPPO

GAME = "MortalKombatII-Genesis"
STATE = "LiuKangVsShaoKahn_VeryHard_15"
TRAIN_STEPS = 50_000

def make_env(frame_skip, stickprob):
    def _thunk():
        env = retro.make(
            GAME,
            state=STATE,
            use_restricted_actions=Actions.DISCRETE
        )
        env = StochasticFrameSkip(env, n=frame_skip, stickprob=stickprob)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        return env
    return _thunk

def objective(trial):
    # Sample HPPO hyperparameters
    manager_lr      = trial.suggest_float("manager_lr",     1e-5, 1e-3, log=True)
    worker_lr       = trial.suggest_float("worker_lr",      1e-5, 1e-3, log=True)
    clip_ratio      = trial.suggest_float("clip_ratio",     0.1, 0.3)
    gamma           = trial.suggest_float("gamma",          0.90, 0.999)
    gae_lambda      = trial.suggest_float("gae_lambda",     0.8, 1.0)
    option_duration = trial.suggest_int("option_duration", 4, 16)
    steps_per_epoch = trial.suggest_categorical("steps_per_epoch", [512, 1024, 2048])
    train_epochs    = trial.suggest_int("train_epochs",    1, 10)
    n_envs          = trial.suggest_categorical("n_envs", [2, 4, 8])
    frame_skip      = trial.suggest_categorical("frame_skip", [2, 4, 8])
    stickprob       = trial.suggest_float("stickprob",      0.0, 0.5)

    # Build vectorized environments
    envs = SubprocVecEnv([make_env(frame_skip, stickprob)] * n_envs)
    envs = VecFrameStack(envs, n_stack=4)
    envs = VecTransposeImage(envs)

    # Instantiate HPPO
    model = HPPO(
        envs,
        manager_lr=manager_lr,
        worker_lr=worker_lr,
        clip_ratio=clip_ratio,
        gamma=gamma,
        gae_lambda=gae_lambda,
        option_duration=option_duration,
        steps_per_epoch=steps_per_epoch,
        train_epochs=train_epochs,
        callback=None
    )

    # Train for 50k steps (multiple epochs = TRAIN_STEPS // steps_per_epoch)
    model.learn(TRAIN_STEPS)

    #extract the last reward
    if not model.reward_history or np.isnan(model.reward_history[-1]):
        return -np.inf
    return float(model.reward_history[-1])

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=20)

    print("Best trial parameters:")
    print(study.best_trial.params)
