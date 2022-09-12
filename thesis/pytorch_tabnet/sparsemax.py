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
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin (새로운 함수 정의시 필요)
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax
        Returns
        -------
        output : torch.Tensor
            same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        # https://medium.com/deeplearningmadeeasy/sparsemax-from-paper-to-code-351e9b26647b 여기 참고
        output = torch.clamp(input - tau, min=0) # 최소값, 최대값 지정해주면 그에 맞게 mapping
        ctx.save_for_backward(supp_size, output)
        # forward() 함수 내부에서 텐서들을 저장해주어야 backward()함수에서 저장된 텐서들을 불러올 수 있음
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        # 저장된 텐서 불러옴.
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        # 0이 아닌 출력에 대해서만 softmax 함수와 같이 back propagation 진행
        return grad_input, None
    
    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold
        Sparsemax는 tau를 활용하여 숫자들을 조정한다. 따라서 tau를 구하는 과정이 필수다.
        
        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax
        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor
        """
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum
        
        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1)
        tau /= support_size.to(input.dtype)
        return tau, support_size

sparsemax = SparsemaxFunction.apply

class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()
    
    def forward(self, input):
        return sparsemax(input, self.dim)

class Entmax15Function(Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        input = input / 2
        
        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = torch.clamp(input - tau_star, min = 0)**2
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        Y, _= ctx.saved_tensors
        gppr = Y.sqrt()
        dX = grad_output * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dx -= q * gppr
        return dX, None
    
    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)
        
        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(dim) / rho
        mean_sq = (Xsrt**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean ** 2)
        delta = (1 - ss) / rho
        
        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        return tau_star, support_size
    
class Entmoid15(Function):
    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        return torch.where(is_pos, 1 - y_neg, y_neg)
    
    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input

entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply

class Entmax15(nn.Module):

    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)