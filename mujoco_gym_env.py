import os
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import mujoco
import numpy as np
from envs.mujoco_env import RenderingConfig
import gymnasium as gym
from gymnasium import spaces

class PandaArm(gym.Env):
    def __init__(
        self,
        xml_path: Path,
        seed: int = None,
        physics_dt: float = 0.1,
        control_dt: float = 0.1,
        time_limit: float = float("inf"),
        rendering_config: Optional[RenderingConfig] = None,
        max_timesteps = 2000,
        render_mode: str = "rgb_array",
    ):
        self.render_mode = render_mode
        if rendering_config is None:
            rendering_config = RenderingConfig()
        xml_path = Path(__file__).parent / xml_path
        with open(xml_path.as_posix(), "r") as file:
            environment_xml = file.read()
        

        os.chdir(xml_path.parent.as_posix()) # clumsy but works
        self._model = mujoco.MjModel.from_xml_string(environment_xml)
        os.chdir(Path(__file__).parent.as_posix())

        self._data = mujoco.MjData(self._model)

        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)

        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._rendering_config = rendering_config

        self._viewer: Optional[mujoco.Renderer] = None
        self._scene_option = mujoco.MjvOption()
        self._info = {}


        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera_id =  0 #mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "chriscamera")
        self.camera.fixedcamid = self.camera_id
        

        self.max_timesteps = max_timesteps
        self.timesteps = 0

        action_space_low = np.array([-2.9, -1.76, -2.9, -3, -2.9, 0, -2.9, 0])
        action_space_high = np.array([2.9, 1.76, 2.9, 0, 2.9, 3.75, 2.9, 255])

        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, shape=(8,), dtype=np.float32)


        self.observation_space = spaces.Box(low = 0, high = 255, shape = (21,), dtype = np.float32)

    
    def done(self, cube_pos, end_effector, timesteps):
        return np.linalg.norm(cube_pos-end_effector) < 0.04 or timesteps >= self.max_timesteps


    def reward(self, cube_pos, end_effector):
        return -np.linalg.norm(cube_pos-end_effector)

    
    def reset(self, seed = None):
        #Reset Box Pos
        cube_pos = np.random.rand(3,)

        self._data.geom_xpos[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "cube")] = cube_pos

        cube_pos = self._data.geom_xpos[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "cube")] 

        self._data.qpos = np.zeros_like(self._data.qpos)
        self._data.qvel = np.zeros_like(self._data.qvel)

        self.timesteps = 0

        state = []
        for i in range(self._model.njnt):
            state.append(self._data.qpos[i])
            state.append(self._data.qvel[i])
        [state.append(x) for x in cube_pos]
        state = np.array(state)

        return state, {}
    
    def render(
        self,
        depth: bool = False,
        segmentation: bool = False,
        scene_option: Optional[mujoco.MjvOption] = None,
        scene_callback: Optional[Callable[[mujoco.MjvScene], None]] = None,
    ) -> np.ndarray:
        """Render a camera view as a numpy array of pixels.

        Args:
            depth: If `True`, this method returns a NumPy float array of depth values
                (in meters). Defaults to `False`, which results in an RGB image.
            segmentation: If `True`, this method returns a 2-channel NumPy int32 array
                of label values where the pixels of each object are labeled with the
                pair (mjModel ID, mjtObj enum object type). Background pixels are
                labeled (-1, -1). Defaults to `False`, which returns an RGB image.
            scene_option: An optional `mujoco.MjvOption` instance that can be used to
                render the scene with custom visualization options. If `None`, the
                default `mujoco.MjvOption` instance is used.
            scene_callback: Optional callback to modify the scene after it has been
                created but before it is rendered. Must accept a single argument of
                type `mujoco.MjvScene`. By default, this is `None`.

        Returns:
            - If `depth` and `segmentation` are both `False`, a (height, width, 3) array
                of uint8 RGB values.
            - If `depth` is `True`, a (height, width) array of float32 depth values in
                meters.
            - If `segmentation` is True, this is a (height, width, 2) int32 numpy
                array where the first channel contains the integer ID of the object at
                each pixel, and the second channel contains the corresponding object
                type (a value in the `mjtObj` enum). Background pixels are labeled
                (-1, -1).
        """

        # if self._viewer is None:
        #     self._viewer = mujoco.Renderer(
        #         model=self._model,
        #         height=self._rendering_config.height,
        #         width=self._rendering_config.width,
        #     )

        with mujoco.Renderer(
                model=self._model,
                height=self._rendering_config.height,
                width=self._rendering_config.width,
            ) as viewer:
            if depth and segmentation:
                raise ValueError("Only one of depth or segmentation can be enabled.")
            if depth:
                viewer.enable_depth_rendering()
            elif segmentation:
                viewer.enable_segmentation_rendering()
            else:
                viewer.disable_depth_rendering()
                viewer.disable_segmentation_rendering()
            scene_option = scene_option or self._scene_option


            viewer.update_scene(
                self._data, self._rendering_config.camera_id, scene_option
            )
            if scene_callback is not None:
                scene_callback(viewer.scene)
            return viewer.render()
    
    def rescale_action(self, action):
        scale = np.array([2.9, 1.76, 2.9, 3/2, 2.9, 3.75/2, 2.9, 255/2])
        offset = np.array([0, 0, 0, -3/2, 0, 3.75/2, 0, 255/2])
        action = scale * action + offset
#        action = action * np.array([1, 1, 1, 1, 1, 1, 1, 0])
        return action/5
    
    def step(self, action: np.ndarray) -> NamedTuple:
        # Rescale actions to proper setpoint values
        action = self.rescale_action(action)

        self._data.ctrl[:] = action
        mujoco.mj_step(self._model, self._data)
        model = self._model
        data = self._data

        cube_pos = self._data.geom_xpos[mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "cube")]

        finger1 = self._data.xpos[-1]
        finger2 = self._data.xpos[-2]
        end_effector = (finger1 + finger2) / 2

        # building state
        state = []
        for i in range(self._model.njnt):
            state.append(self._data.qpos[i])
            state.append(self._data.qvel[i])
        [state.append(x) for x in cube_pos]

        state = np.array(state)

        # if np.linalg.norm(cube_pos-end_effector) < 0.02 or self.timesteps == self.max_timesteps:
        #     reward = -np.linalg.norm(cube_pos-end_effector)*(self.max_timesteps - self.timesteps)
        #     return dm_env.termination(reward = reward, observation=state)
        # else:
        #     reward = -np.linalg.norm(cube_pos-end_effector)
        #     self.timesteps += 1
        #     return dm_env.transition(reward = reward, observation = state)
        self.timesteps += 1
        done = self.done(cube_pos, end_effector, self.timesteps)
        reward = self.reward(cube_pos, end_effector)

        if done:
            reward *= (self.max_timesteps - self.timesteps)
        
        return (
            state,
            reward,
            done,
            False,# TODO: Still figure this bitch out
            {},
        )


