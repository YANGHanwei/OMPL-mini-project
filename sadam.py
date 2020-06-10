import torch
from torch.optim.optimizer import Optimizer

def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


class SAdam(Optimizer):
    r"""Implements SAdam algorithm.

    It has been proposed in `SAdam: A Variant of Adam for Strongly Convex Functions`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _SAdam\: A Method for Stochastic Optimization:
        https://openreview.net/pdf?id=rye5YaEtPr
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, xi_1=0.1, xi_2=1,
                 gamma=0.9, weight_decay=0, delta=1e-2, vary_delta=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= delta:
            raise ValueError("Invalid delta parameter: {}".format(delta))            
        if not 0.0 <= gamma:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))  
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, beta1=beta1, 
                        weight_decay=weight_decay, delta=delta,
                        vary_delta=vary_delta, gamma=gamma,
                        xi_1=xi_1, xi_2=xi_2)
        super(SAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('gamma', 0.9)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad # Step 4 of paper
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['vs'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['gs'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                vs, gs = state['vs'], state['gs']
                beta1= group['beta1']
                
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                gs.mul_(beta1).add_(grad, alpha=1 - beta1) #Step 5 of paper
                vs.mul_(1-(group['gamma'] / state['step'])).addcmul_(grad, grad, value=group['gamma'] / state['step']) #Step 6 of paper
#                 print("VS=",vs.size(), "EYE=", eye_like(vs).size())
                if len(list(vs.size())):
                    vs.add_(eye_like(vs), alpha=group['delta'] / state['step']) #Step 7 of paper
                else:
                    vs.add_(1, alpha=group['delta'] / state['step'])
                
                step_size = group['lr'] / state['step']

                p.addcdiv_(gs, vs, value=-step_size) #Step 8 of paper

        return loss