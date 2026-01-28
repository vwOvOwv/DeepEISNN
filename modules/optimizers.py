import torch
from torch.optim.optimizer import Optimizer


class EiSGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, decay_mode='L2', nesterov=False, clamped=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, decay_mode=decay_mode,
                        nesterov=nesterov,
                        clamped=clamped)
        
        super(EiSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(EiSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):   # type: ignore
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            decay_mode = group['decay_mode']
            nesterov = group['nesterov']
            clamped = group['clamped']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                if weight_decay != 0:
                    if decay_mode == 'L2':
                        d_p = d_p.add(p, alpha=weight_decay)
                    elif decay_mode == 'L1':
                        d_p = d_p.add(torch.sign(p), alpha=weight_decay)

                if momentum != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.clone(d_p).detach()
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                p.add_(d_p, alpha=-lr)
                if clamped:
                    p.data.clamp_(min=0.0)

        return loss