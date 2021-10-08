import gym
import pybullet_envs as pe

from stable_baselines3 import PPO

# import stable_baselines3

# from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import time
from torch.nn import ReLU, Tanh

import pyhopper


def score_trained_model(model):
    final_rewards = []
    for i in range(20):
        obs = model.env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = model.env.step(action)
            total_reward += reward
        final_rewards.append(total_reward)
    return np.mean(final_rewards)


def score_random():
    env = gym.make("CartPoleContinuousBulletEnv-v0")

    final_rewards = []
    for i in range(20):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
        print("steps: ", steps)
        final_rewards.append(total_reward)
    return np.mean(final_rewards)


def render_trained_model(model):
    model.env.render()
    obs = model.env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = model.env.step(action)
        # model.env.render()
    model.env.close()


def train_ppo(params, render=False):
    env = gym.make("CartPoleContinuousBulletEnv-v0")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": [
                    {
                        "pi": [params["size"], params["size"]],
                        "vf": [params["size"], params["size"]],
                    }
                ],
                "activation_fn": {"relu": ReLU, "tanh": Tanh}[params["activation"]],
                "normalize_images": True,
            },
            learning_rate=params["lr"],
            n_epochs=params["n_epochs"],
            gae_lambda=params["gae_lambda"],
            max_grad_norm=params["max_grad_norm"],
            ent_coef=params["ent_coef"],
        ).learn(40000)
    except (ValueError, ZeroDivisionError):
        raise pyhopper.CancelEvaluation()
    # score, _ = evaluate_policy(model, gym.make("AntBulletEnv-v0"))
    # print("total reward: ", score)
    score = score_trained_model(model)
    # print("total reward: ", score)
    if render:
        print("total reward: ", score)
        render_trained_model(model)
    return score


if __name__ == "__main__":
    print("Random", score_random())
    # default_params = {
    #     "lr": 0.0005,
    #     "n_epochs": 10,
    #     "gae_lambda": 0.95,
    #     "max_grad_norm": 1,
    #     "ent_coef": 0,
    #     "activation": "tanh",
    #     "size": 128,
    # }
    # start = time.time()
    # train_ppo(default_params)
    # took = time.time() - start
    # print(f"Took {took/60:0.1f} minutes")
    # import sys

    # sys.exit(1)
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(0.005, 0.0001, log=True),
            "n_epochs": pyhopper.int(5, 20),
            "gae_lambda": pyhopper.choice([0.8, 0.9, 0.95, 0.99], is_ordinal=True),
            "max_grad_norm": pyhopper.choice([0.1, 0.5, 1.0, 2.0], is_ordinal=True),
            "ent_coef": pyhopper.float(0, 0.2, precision=1),
            "activation": pyhopper.choice(["tanh", "relu"]),
            "size": pyhopper.int(64, 256, power_of=2),
        }
    )
    best_params = search.run(
        pyhopper.wrap_n_times(train_ppo, n=3, yield_after=0),
        # train_ppo,
        direction="max",
        timeout="4h",
        n_jobs="4x per-gpu",
        canceler=pyhopper.cancelers.QuantileCanceler(0.6),
    )
    print("best_params", best_params)
    train_ppo(best_params, render=True)

# ============================ Summary ===========================
# Mode              : Best f : Steps : Canceled : Time
# ----------------  : ----   : ----  : ----     : ----
# Initial solution  : 200    : 1     : 0        : 08:32 (m:s)
# Random seeding    : 200    : 32    : 16       : 03:32:42 (h:m:s)
# Local sampling    : 200    : 118   : 19       : 10:32:28 (h:m:s)
# ----------------  : ----   : ----  : ----     : ----
# Total             : 200    : 151   : 35       : 03:57:31 (h:m:s)
# ================================================================
# best_params {'lr': 0.0017706780889095847, 'n_epochs': 8, 'gae_lambda': 0.9, 'max_grad_norm': 2.0, 'ent_coef': 0.1, 'activation': 'tanh', 'size': 64}