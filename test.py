from a2c_continuous import ActorCritic
from envs.mujoco_env import MujocoEnv
import numpy as np



env = MujocoEnv("/home/aadia/nmepproject/envs/xmls/scene.xml", max_timesteps=1000)
#env = MujocoEnv("/home/aadia/nmepproject/envs/xmls/panda.xml", max_timesteps=1000)

model = ActorCritic(np.zeros((21,)), np.zeros((8,)), "model")


# model.load_state_dict()
model.train(env,
            20000002)