import torch
from torch import nn

from collections import OrderedDict


class ReLUNetwork(nn.Module):

    def __init__(self, width):
        super(ReLUNetwork, self).__init__()

        self.depth = len(width)
        self.width = width
        self.fcs = list()

        self.net = self._build_network(width)

    def _build_network(self, width):
        network = OrderedDict()
        current_layer = 1
        for input_size, output_size in zip(width, width[1:]):
            if current_layer > 1:
                network[str(current_layer)] = nn.ReLU()
                current_layer += 1
            self.fcs.append(nn.Linear(input_size, output_size))
            network[str(current_layer)] = self.fcs[-1]
            current_layer += 1
        return nn.Sequential(network)

    def input_config(self, x, return_outputs=True):
        outputs = self.forward_layer(x)
        config = [(output.squeeze() > 0).type(torch.float32) for output in outputs]
        if return_outputs:
            return outputs, config
        return config

    def get_polytope(self, x, as_tensor=False):
        configs = self.input_config(x, False)
        return self._get_polytope_from_configs(configs, as_tensor)

    def _get_polytope_from_configs(self, configs, as_tensor=False):

        lambdas = [torch.diag(config) for config in configs]
        j_s = [torch.diag(-2 * config + 1) for config in configs]

        w_s = [self.fcs[0].weight]
        b_s = [self.fcs[0].bias]

        for (i, fc) in enumerate(self.fcs[1:]):
            current_ws = w_s[-1]
            current_bs = b_s[-1]
            lambda_i = lambdas[i]
            step = fc.weight.matmul(lambda_i)
            w_s.append(step.matmul(current_ws))
            b_s.append(step.matmul(current_bs) + fc.bias)

        a_stack = []
        b_stack = []
        for j, wk, bk in zip(j_s, w_s, b_s):
            a_stack.append(j.matmul(wk))
            b_stack.append(-j.matmul(bk))
        if as_tensor:
            return {'a_stack': a_stack,
                    'b_stack': b_stack,
                    'total_a': w_s[-1],
                    'total_b': b_s[-1]}

        polytope_A = torch.cat(a_stack, dim=0).cpu().detach().numpy()
        polytope_b = torch.cat(b_stack, dim=0).cpu().detach().numpy()

        return {'poly_a': polytope_A,
            'poly_b': polytope_b,
            'configs': configs,
            'total_a': w_s[-1],
            'total_b': b_s[-1]
            }

    def forward_layer(self, x):
        if x.shape[-1] != self.width[0]:
            x = x.view(-1, self.width[0])
        values = list()
        for fc in self.fcs[:-1]:
            x = fc(x)
            values.append(x.clone())
            x = nn.functional.relu(x)
        return values

    def forward(self, x):
        x = self.net(x)
        return x

