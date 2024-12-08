from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import dm_env
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
import re
import os

@dataclass(frozen=True)
class RenderingConfig:
    """Camera options for rendering.

    Attributes:
        height: The height of the rendered image in pixels.
        width: The width of the rendered image in pixels.
        camera_id: The ID of the camera to use for rendering. Can be a string or an
            integer. Defaults to -1, which uses the free camera.
    """

    height: int = 1920
    width: int = 1080
    camera_id: str | int = -1


class MujocoEnv(dm_env.Environment):
    """A MuJoCo environment with a dm_env.Environment interface.

    Subclasses should implement the following methods:
    - reset(self) -> dm_env.TimeStep
    - step(self, action) -> TimeStep
    - observation_spec(self)
    - action_spec(self)
    """

    def __init__(
        self,
        xml_path: Path,
        seed: int = None,
        physics_dt: float = 0.1,
        control_dt: float = 0.1,
        time_limit: float = float("inf"),
        rendering_config: Optional[RenderingConfig] = None,
    ):
        """Initializes a new MujocoEnv.

        Args:
            xml_path: Path to the MuJoCo XML file.
            seed: Seed for the random number generator.
            physics_dt: The physics timestep.
            control_dt: The control timestep. Must be a multiple of physics_dt.
            time_limit: The maximum duration of an episode. By default, this is set to
                infinity.
            rendering_config: A `RenderingConfig` instance that specifies the camera
                options for rendering.
        """
        if rendering_config is None:
            rendering_config = RenderingConfig()
        xml_path = Path(__file__).parent / xml_path
        with open(xml_path.as_posix(), "r") as file:
            environment_xml = file.read()
        

        os.chdir(xml_path.parent.as_posix()) # clumsy but works
        self._model = mujoco.MjModel.from_xml_string(environment_xml)
        os.chdir(Path(__file__).parent.as_posix())

        self._data = mujoco.MjData(self._model)

        for i in range(len(self._Kq)):
            actuator_name = f"actuator{i+1}"
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self._model.actuator_gainprm[actuator_id, 0] = self._Kq[i]
            self._model.actuator_biasprm[actuator_id, 0] = self._Kqd[i]


        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)

        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._rendering_config = rendering_config

        self._viewer: Optional[mujoco.Renderer] = None
        self._scene_option = mujoco.MjvOption()
        self._info = {}

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
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._rendering_config.height,
                width=self._rendering_config.width,
            )
        if depth and segmentation:
            raise ValueError("Only one of depth or segmentation can be enabled.")
        if depth:
            self._viewer.enable_depth_rendering()
        elif segmentation:
            self._viewer.enable_segmentation_rendering()
        else:
            self._viewer.disable_depth_rendering()
            self._viewer.disable_segmentation_rendering()
        scene_option = scene_option or self._scene_option
        self._viewer.update_scene(
            self._data, self._rendering_config.camera_id, scene_option
        )
        if scene_callback is not None:
            scene_callback(self._viewer.scene)
        return self._viewer.render()

    def close(self) -> None:
        """Clean up resources associated with the environment."""
        if self._viewer is not None:
            self._viewer.close()
        self._viewer = None

    def time_limit_exceeded(self) -> bool:
        """Returns True if the simulation time has exceeded the time limit."""
        return self._data.time >= self._time_limit

    # Convenience methods.

    def forward(self) -> None:
        mujoco.mj_forward(self._model, self._data)
    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def time_limit(self) -> float:
        return self._time_limit

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random

    @property
    def scene_option(self) -> mujoco.MjvOption:
        return self._scene_option

    @property
    def info(self) -> dict:
        return self._info
