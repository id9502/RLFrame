import torch
from torch import nn


class NN(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        super(NN, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _reshape_output(self, x):
        return x.view(-1, *self.output_shape)

    def get_flat_params(self) -> torch.Tensor:
        flat_params = torch.cat([params.view(-1) for params in self.parameters()])
        return flat_params

    def set_flat_params(self, flat_params) -> None:
        flat_params = torch.as_tensor(flat_params, dtype=self.parameters().__next__().dtype)
        prev_ind = 0
        for param in self.parameters():
            flat_size = param.nelement()
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def get_flat_grad(self, grad_grad: bool = False) -> torch.Tensor:
        flat_grad = torch.cat([param.grad.grad.view(-1) if grad_grad else param.grad.view(-1) for param in self.parameters()])
        return flat_grad
