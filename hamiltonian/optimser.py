import numpy as np
import torch


class hamiltonian(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=1e-6):
        default = dict(lr=lr, eps=eps)
        super().__init__(params, default)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['momentum'] -= self.defaults['eps']*grad/2 + state['momentum']*self.defaults['lr']
                    sqsum = np.vdot(state['momentum'], state['momentum'])
                    p.data += self.defaults['eps'] * state['momentum']/(1 + sqsum)


