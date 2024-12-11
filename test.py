
import imageio
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import tqdm

from mujoco_gym_env import PandaArm

xml_path = "/home/aadia/nmepproject/envs/xmls/scene.xml"
env = PandaArm(xml_path=xml_path,
                max_timesteps=1000,
               )



model = PPO.load("ppo_panda_574")


# Test


vec_env = make_vec_env(PandaArm, n_envs=1, env_kwargs=dict(xml_path=xml_path, max_timesteps=1000))
obs = vec_env.reset()
n_steps = 100

frames = []
pbar = tqdm.tqdm(range(n_steps))
for step in pbar:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    frames.append(vec_env.render())
    pbar.set_description(f"Reward {reward}")


    if done:
        break

print(len(frames))

frames = [frame for frame in frames if frame is not None]
imageio.mimsave("frames.mp4", frames, fps=30)
