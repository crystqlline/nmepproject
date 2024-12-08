import time
import mujoco
import mujoco.viewer

def main():
    """
    Main function to launch a MuJoCo simulation viewer.

    The function launches a viewer to visualize the simulation. This is mainly used for debugging simulator environment.
    The viewer will automatically close after 10000 seconds.

    usage:  Linux/Windows etc: run `python render.py`
            MacOS: run `mjpython render.py`

    The initial joint positions are matching the initial configuration of the real setup.

    Returns:
        None
    """
    m = mujoco.MjModel.from_xml_path("xmls/scene.xml")
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        time_step = 0
        while viewer.is_running():
            step_start = time.time()
            time_step += 1
            mujoco.mj_step(m, d)
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
