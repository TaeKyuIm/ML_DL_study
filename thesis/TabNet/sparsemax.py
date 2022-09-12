from turtle import forward
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import torch

def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d+1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)

class SparsemaxFunction(Function):
    """
    Sparsemax Function을 실행하는 클래스
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0) # 최소값, 최대값 지정해주면 그에 맞게 mapping
        ctx.save_for_backward(supp_size, output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None
    