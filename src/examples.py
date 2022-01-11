import torch
from torch import optim, nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src._polytope_ import Polytope
from src.models import relu_network
from src.utilities import plot_polytopes_2d


def example_2d(point=None):
    # Training block -- train a simple 2d Network
    # --- define network
    plnn_obj = relu_network.ReLUNetwork([2, 20, 50, 20, 2])
    net = plnn_obj

    # --- define random training points
    m = 12
    np.random.seed(3)
    x = [np.random.uniform(size=2)]
    r = 0.16
    while len(x) < m:
        p = np.random.uniform(size=2)
        if min(np.abs(p - a).sum() for a in x) > 2 * r:
            x.append(p)
    epsilon = r / 2
    X = torch.Tensor(np.array(x))
    torch.manual_seed(1)
    y = (torch.rand(m) + 0.5).long()

    # --- train network
    opt = optim.Adam(net.parameters(), lr=1e-3)
    for i in range(1000):
        out = net(Variable(X))
        loss = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1].data != y).float().mean()
        if i % 100 == 0:
            print(loss.item(), err)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # --- display network decision boundaries
    XX, YY = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    X0 = Variable(torch.Tensor(np.stack([np.ravel(XX), np.ravel(YY)]).T))
    y0 = net(X0)
    ZZ = (y0[:, 0] - y0[:, 1]).resize(100, 100).data.numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(XX, YY, -ZZ, cmap="coolwarm", levels=np.linspace(-1000, 1000, 3))
    ax.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=y.numpy(), cmap="coolwarm", s=70)
    ax.axis("equal")
    ax.axis([0, 1, 0, 1])

    for a in x:
        ax.add_patch(patches.Rectangle((a[0] - r / 2, a[1] - r / 2), r, r, fill=False))

    if point is not None:
        pol = Polytope.from_polytope_dict(plnn_obj.get_polytope(point), point)
        xylim = [-1, 1]
        plt.xlim(xylim[0], xylim[1])
        plt.ylim(xylim[0], xylim[1])
        plot_polytopes_2d([pol], xylim=xylim, input_point=point.numpy())
        plt.show()
