import optuna
import retro
from retro import Actions
from retros import StochasticFrameSkip 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from ppo import PPO

GAME = "MortalKombatII-Genesis"
STATE = "LiuKangVsShaoKahn_VeryHard_15"
TRAIN_STEPS = 50_000

def objective(trial):
    # 1) Sample hyperparameters
    lr     = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    clr    = trial.suggest_float("clr", 1e-5, 1e-3, log=True)
    eps    = trial.suggest_float("eps", 0.05, 0.3)
    lreps  = trial.suggest_float("lreps", 1e-7, 1e-3, log=True)
    gamma  = trial.suggest_float("gamma", 0.90, 0.999)
    steps_batch = trial.suggest_categorical("steps_batch", [824])
    updates_per_iteration = trial.suggest_int("updates_per_iteration", 1, 8)
    n_envs = trial.suggest_categorical("n_envs", [4])
    frame_skip = trial.suggest_categorical("frame_skip", [4])
    stickprob  = trial.suggest_float("stickprob", 0.0, 0.5)

    # 2) Build a factory that closes over frame_skip & stickprob
    def make_thunk():
        env = retro.make(
            GAME,
            state=STATE,
            scenario=None,
            use_restricted_actions=Actions.DISCRETE
        )
        # apply custom wrapper with sampled params
        env = StochasticFrameSkip(env, n=frame_skip, stickprob=stickprob)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        return env

    # 3) Create the vectorized envs
    envs = SubprocVecEnv([make_thunk] * n_envs)
    envs = VecFrameStack(envs, n_stack=4)
    envs = VecTransposeImage(envs)

    # 4) Instantiate PPO with sampled hyperparameters
    model = PPO(
        envs,
        callback=None,
        lr=lr,
        clr=clr,
        eps=eps,
        lreps=lreps,
        gamma=gamma,
        steps_batch=steps_batch,
        steps_episode=steps_batch,
        updates_per_iteration=updates_per_iteration
    )

    # 5) Train and report final mean reward
    model.learn(steps=TRAIN_STEPS)
    return model.reward_history[-1]

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial.params)
