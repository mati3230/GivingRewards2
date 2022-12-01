import gym
import torch as th
import os
import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
#from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
#from stable_baselines3.common.env_util import make_vec_env
from superpoint_growing_env import SuperpointGrowingEnv


def main():
    # Create environment
    dataset = "pcg"
    alg = "ppo"
    env = SuperpointGrowingEnv(mapping=[4,2,1], ignore=[1,2], dataset=dataset,
        density_method="rel", obj_punish=0.1)

    # very slow!
    #num_cpu = 4
    #env = make_vec_env(SuperpointGrowingEnv, n_envs=num_cpu, vec_env_cls=DummyVecEnv)

    net_arch = [8,8,4]
    if dataset == "pcg":
        net_arch = [7,7,4]

    # Instantiate the agent
    if alg == "dqn":
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
            net_arch=net_arch)
        model = DQN("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, 
            tensorboard_log="./logs/" + dataset + "/drl",
            learning_rate=1e-4, seed=42, exploration_fraction=0.5)
    else:
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
            net_arch=[dict(pi=net_arch, vf=net_arch)])
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, 
            tensorboard_log="./logs/" + dataset + "/drl",
            learning_rate=1e-3, seed=42)


    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(env, best_model_save_path="./models/drl/"+dataset+"/",
        log_path="./models/drl/"+dataset+"/", eval_freq=2e4,
        deterministic=True, render=False)

    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(6e6), progress_bar=True, callback=eval_callback)


if __name__ == "__main__":
    main()