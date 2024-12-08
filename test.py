from a2c_continuous import ActorCritic
from envs.mujoco_env import MujocoEnv
import numpy as np



env = MujocoEnv("/home/aadia/nmepproject/envs/xmls/scene.xml")
model = ActorCritic(np.zeros((16,)), np.zeros((8,)), "model")


model.train(env, 20002)