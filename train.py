
import imageio
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

from mujoco_gym_env import PandaArm

xml_path = "/home/aadia/nmepproject/envs/xmls/scene.xml"
checkpoint_path = "ppo_panda_574"

training_timesteps = 300000

env = PandaArm(xml_path=xml_path,
                max_timesteps=1000,
               )

if checkpoint_path is not None:
    model = PPO.load(checkpoint_path, env = env)
else:
    model = PPO("MlpPolicy", env, verbose=1, device = "cuda")

model = model.learn(total_timesteps=training_timesteps)

model.save(f"ppo_panda_{np.random.randint(10000000000)}")


# Test


vec_env = make_vec_env(PandaArm, n_envs=1, env_kwargs=dict(xml_path=xml_path, max_timesteps=1000))
obs = vec_env.reset()
n_steps = 200

frames = []
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    frames.append(vec_env.render())

    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

print(len(frames))

frames = [frame for frame in frames if frame is not None]
imageio.mimsave("frames.mp4", frames, fps=30)
