import torch
from core.model.policy import Policy
from core.model.nets.mlp_critic import ValueNet
from core.model.nets.mlp_policy import PolicyNet
from core.common import StepDictList, ParamDict
from core.utilities.convenience import cpu_state_dict


__all__ = ["PolicyWithValue"]


class PolicyWithValue(Policy):
    """
    This policy only handles low dimension information from observations when estimating 'a' and 'v'
    """

    def __init__(self, config_dict: ParamDict, env_info: dict):
        # require check failed mainly because you have not passed config to environment first,
        # which will fill-in action dim and state dim fields
        assert "state dim" in env_info and "action dim" in env_info,\
            f"Error: Key 'state dim' or 'action dim' not in env_info, which only contains {env_info.keys()}"
        super(PolicyWithValue, self).__init__()

        self.value_net = None
        self.policy_net = None

        # TODO: do some actual work here
        self._action_dim = env_info["action dim"]
        self._state_dim = env_info["state dim"]
        self.is_fixed = False

    def init(self):
        self.policy_net = PolicyNet(self._state_dim, self._action_dim)
        self.policy_net.to(self.device)
        self.value_net = ValueNet(self._state_dim)
        self.value_net.to(self.device)

    def finalize(self):
        self.policy_net = None
        self.value_net = None

    def step(self, current_step_list: StepDictList) -> StepDictList:
        state = [cs['s'] for cs in current_step_list]
        if isinstance(state[0], torch.Tensor):
            state = torch.stack(state, dim=0).to(dtype=torch.float32, device=self.device)
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        value = self.value_net(state)
        action = self.policy_net.select_action(state, deterministic=self.is_fixed)
        for i, cs in enumerate(current_step_list):
            cs['a'] = action[i].detach().cpu().numpy()
            cs['v'] = value[i].item()
        return current_step_list

    def reset(self, param: ParamDict):
        assert self.policy_net is not None and self.value_net is not None, "Error: you should call init before reset !"
        value, policy = param.require("value state_dict", "policy state_dict")
        if "fixed policy" in param:
            self.is_fixed = param.require("fixed policy")

        self.value_net.load_state_dict(value)
        self.policy_net.load_state_dict(policy)

    def getStateDict(self) -> ParamDict:
        return ParamDict({"value state_dict": cpu_state_dict(self.value_net.state_dict()),
                          "policy state_dict": cpu_state_dict(self.policy_net.state_dict())})

    def to_device(self, device: torch.device):
        super().to_device(device)
        if self.policy_net is not None:
            self.policy_net.to(device)
        if self.value_net is not None:
            self.value_net.to(device)
