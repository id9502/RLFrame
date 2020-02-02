from core.agent.agent import Agent


class MultiAgent(Agent):
    """
    An agent class will only maintain one policy net but multiple environments,
    useful for most of multi-agent RL/IL settings
    """
    def __init__(self):
        super(MultiAgent, self).__init__()
        assert False, "This class is in developing, do not use it"

    def broadcast(self, config):
        pass

    def collect(self):
        pass

    @staticmethod
    def __sampler():
        pass
