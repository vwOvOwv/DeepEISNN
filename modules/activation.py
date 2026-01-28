import torch
from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import surrogate
from typing import Union

surrogate_dict = {
    'sigmoid': surrogate.Sigmoid,
    'atan': surrogate.ATan,
    'leaky_relu': surrogate.LeakyKReLU,
}

class LIF(LIFNode):
    def __init__(self, tau: float = 2.0, surrogate_function: str = 'sigmoid', 
                 step_mode: str = 'm', decay_input: bool = False, 
                 v_threshold: float = 1.0, v_reset: float = None, 
                 detach_reset: bool = True):

        super().__init__(float(tau), decay_input, v_threshold, v_reset, 
                         surrogate_function=surrogate_dict[surrogate_function](),
                         detach_reset=detach_reset, step_mode=step_mode)
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, None]:
        spike = super().forward(x)
        return spike
        