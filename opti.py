import torch
from torch.optim.optimizer import Optimizer

def eye_like(tensor):
    """ Return an eye tensor corresponding to the tensor size. """
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


class SAdam(Optimizer):
    r"""Implements SAdam algorithm.

    It has been proposed in `SAdam: A Variant of Adam for Strongly Convex Functions`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta1 (float, optional): coefficient used for computing
            running average of gradient  (default: 0.9)
        gamma (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        delta (float, optional): used to improve convergence
        

    .. _SAdam\: A Method for Stochastic Optimization:
        https://openreview.net/pdf?id=rye5YaEtPr
    """

    def __init__(self, params, lr=1e-3, beta1=0.9,
                 gamma=0.9, weight_decay=0, delta=1e-2):
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
                        gamma=gamma)
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
                if len(list(vs.size())) > 1:
                    vs.add_(eye_like(vs), alpha=group['delta'] / state['step']) #Step 7 of paper
                else:
                    vs.add_(1, alpha=group['delta'] / state['step'])
                
                step_size = group['lr'] / state['step']

                p.addcdiv_(gs, vs, value=-step_size) #Step 8 of paper

        return loss
    
class Signum(Optimizer):
    # Ref: https://github.com/jiaweizzhao/Signum_pytorch
    r"""Implements Signum optimizer that takes the sign of gradient or momentum.

    See details in the original paper at:https://arxiv.org/abs/1711.05101

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay (default: 0)

    Example:
        >>> optimizer = signum.Signum(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. note::
        The optimizer updates the weight by:
            buf = momentum * buf + (1-momentum)*rescaled_grad
            weight = (1 - lr * weight_decay) * weight - lr * sign(buf)

        Considering the specific case of Momentum, the update Signum can be written as

        .. math::
                \begin{split}g_t = \nabla J(W_{t-1})\\
                 m_t = \beta m_{t-1} + (1 - \beta) g_t\\
                W_t = W_{t-1} - \eta_t \text{sign}(m_t)}\end{split}

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        If do not consider Momentum, the update Sigsgd can be written as

        .. math::
                g_t = \nabla J(W_{t-1})\\
                W_t = W_{t-1} - \eta_t \text{sign}(g_t)}

    """
    def __init__(self, params, lr=0.01, momentum=0.09, weight_decay = 0, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)

        super(Signum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Signum, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    # signum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)

                    else:
                        buf = param_state['momentum_buffer']

                    buf.mul_(momentum).add_((1 - momentum),d_p)
                    d_p = torch.sign(buf)
                else:#signsgd
                    d_p = torch.sign(d_p)

                p.data.add_(-group['lr'], d_p)

        return loss
    

