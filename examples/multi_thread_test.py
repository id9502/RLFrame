from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.tasks import CloseMicrowave

from multiprocessing import Process
import numpy as np


def sub_proc(child):
    action_mode = ActionMode()
    env = Environment(action_mode, headless=child)
    env.launch()
    task = env.get_task(CloseMicrowave)

    descriptions, obs = task.reset()

    for i in range(1000):
        obs, reward, terminate = task.step(np.random.normal(np.zeros(action_mode.action_size)))


if __name__ == "__main__":
    proc1 = Process(target=sub_proc, args=(True,))
    proc1.start()
    sub_proc(False)
