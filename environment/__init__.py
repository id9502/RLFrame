try:
    from .fake_gym.fake_gym import FakeGymEnv as FakeGym
except ImportError:
    print("Cannot import GYM environment, please make sure it is appropriately installed")
try:
    from .fake_rlbench.fake_rlbench import FakeRLBenchEnv as FakeRLBench
except ImportError:
    print("Cannot import RLBench environment, please make sure it is appropriately installed")
