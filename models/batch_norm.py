import logging
import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.function import Function

import copy

dsbn_size = None
class DSBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.main_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.aux_bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, input):
        if self.training:
            input0 = self.main_bn(input[:dsbn_size])
            input1 = self.aux_bn(input[dsbn_size:])
            input = torch.cat((input0, input1), 0)
        else:
            input = self.main_bn(input)
        return input

    @classmethod
    def convert_dsbn(self, module, process_group=None):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = self(module.num_features,
                                 eps=module.eps,
                                 momentum=module.momentum,
                                 affine=module.affine,
                                 track_running_stats=module.track_running_stats)
            if module.affine:
                with torch.no_grad():
                    module_output.main_bn.weight = module.weight
                    module_output.main_bn.bias = module.bias

                    module_output.aux_bn.weight = copy.deepcopy(module.weight)
                    module_output.aux_bn.bias = copy.deepcopy(module.bias)

            module_output.main_bn.running_mean = module.running_mean
            module_output.main_bn.running_var = module.running_var
            module_output.main_bn.num_batches_tracked = module.num_batches_tracked

            module_output.aux_bn.running_mean = copy.deepcopy(module.running_mean)
            module_output.aux_bn.running_var = copy.deepcopy(module.running_var)
            module_output.aux_bn.num_batches_tracked = copy.deepcopy(module.num_batches_tracked)

        for name, child in module.named_children():
            module_output.add_module(name, self.convert_dsbn(child, process_group))
        del module
        return module_output