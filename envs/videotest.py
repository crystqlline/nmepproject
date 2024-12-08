# In your main script or wherever you're using the MujocoEnv
from mujoco_env import MujocoEnv
# Initialize the environment
env = MujocoEnv(xml_path="xmls/panda.xml")

# Capture a 10-second video at 30 fps
env.capture_video(duration=10, fps=30, filename="simulation_video.mp4")