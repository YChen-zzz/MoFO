import torch
from torch.optim.optimizer import Optimizer
import math
import torch.distributed as dist
from torch.optim.optimizer import _dispatch_sqrt


device = 'cuda' if torch.cuda.is_available() else 'cpu'





class AdamW_MoFO(Optimizer):
    def __init__(
        self,
        model=None,
        weight_decay=0.1,
        lr=1,
        beta1=0.9,
        beta2=0.999,
        fraction= 0.03,
        epsilon=1e-8,
    ):

        
        self.model = model
        optim_groups = []
        for name, param in self.model.named_parameters():

            if param.requires_grad:
                # build dictionary here
                dic = {}
                dic["name"] = name
                dic["params"] = param
                if ("norm" in name or "ln_f" in name):
                    dic["weight_decay"] = 0

                else:
                    dic["weight_decay"] = weight_decay
                if "bias" not in name and "norm" not in name:
                    dic["fraction"] = fraction
                optim_groups.append(dic)

        defaults = dict(lr= lr, beta1 = beta1, beta2 = beta2, epsilon = epsilon)

        super(AdamW_MoFO, self).__init__(optim_groups, defaults)

    def step(self,closure=None):
        "update"
        with torch.no_grad():
            for group in self.param_groups:
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                lr = group["lr"]
                epsilon = group["epsilon"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]
                    if "iteration" not in state.keys():
                        state["iteration"] = 0

                    state["iteration"] += 1

                    if group["weight_decay"] != 0:
                        p.data.mul_(1 - lr * group["weight_decay"])
                    grad = grad.to(torch.float32)
                    if "m" not in state.keys():
                        state["m"] = torch.zeros_like(grad)
                    state["m"].lerp_(grad, 1 - beta1)

                    if "v" not in state.keys():
                        state["v"] = torch.zeros_like(grad)

                    state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    bias_correction_1 = 1 - beta1 ** state["iteration"]
                    bias_correction_2 = 1 - beta2 ** state["iteration"]
                    bias_correction_2_sqrt = _dispatch_sqrt(bias_correction_2)  # **0.5
                    stepsize = lr / bias_correction_1

                    h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(
                        epsilon
                    )

                    if 'fraction' in group:
                        k = int(p.numel() * group["fraction"])
                        threshold = torch.topk(state["m"].abs().view(-1), k)[0].min()
                        p.addcdiv_(torch.where(state["m"].abs()>=threshold, state["m"], 0), h, value=-stepsize)
                    else:
                        p.addcdiv_(state["m"], h, value=-stepsize)