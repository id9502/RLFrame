import numpy as np
from core.common import StepDictList


def advantage(step_list: StepDictList, gamma: float, tau: float):
    """
    :param step_list: step_list, should have only one trajectory, each step contains 'r' and 'v'
    :param gamma: float
    :param tau: float
    :return: replay_memory
    """

    prev_return = 0.
    prev_value = 0.
    prev_advantage = 0.

    advantages = np.zeros((len(step_list),), dtype=np.float32)
    for i, step in enumerate(reversed(step_list)):
        step["return"] = step['r'] + gamma * prev_return
        delta = step['r'] + gamma * prev_value - step['v']
        advantages[i] = delta + gamma * tau * prev_advantage

        prev_return = step["return"]
        prev_value = step['v']
        prev_advantage = advantages[i]

    # we find the advantage normalization here is harmful for training performance, thus we removed it
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1.e-10)
    for i, step in enumerate(reversed(step_list)):
        step["advantage"] = advantages[i]


if __name__ == "__main__":
    memory = [{'r': 0.1, 'v': 0.0}, {'r': 0.1, 'v': 0.0}, {'r': 0.1, 'v': 0.0}]
    advantage(memory, 0.7, 0.7)
    print(memory)
