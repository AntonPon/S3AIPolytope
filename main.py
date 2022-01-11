import time

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm
from scipy import optimize, linalg
import numpy as np
from matplotlib import pyplot as plt

from src.models import relu_network
from src._polytope_ import Polytope


PATH_TO_SAVE = '/Users/anton/Documents/projects/UnboundedSets/data'
BATCH_SIZE = 16
NUM_WORKERS = 4
ITERATIONS = 30
RESIZE = 10
OUTPUT_SIZE = 20
HIDDEN_SIZE = [10, 30, 10]
DATASET_TYPE = 'mnist'


def get_dataset(path_to_save, transforms, train=True, d_type='mnist'):
    dataset = None
    if d_type == 'mnist':
        dataset = datasets.MNIST(path_to_save,
                                 train=train,
                                 download=True,
                                 transform=transforms)
    if d_type == 'cifar10':
        dataset = datasets.CIFAR10(path_to_save,
                                   train=train,
                                   download=True,
                                   transform=transforms)
    return dataset


def train_network(model, optimizer, criterion, dataloader):
    total_train_loss = 0.
    model.train()
    for batch in tqdm(dataloader):
        imgs, labels = batch
        imgs = imgs.reshape((imgs.size()[0], -1))
        optimizer.zero_grad()
        result = model(imgs)
        loss = criterion(result, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss / len(dataloader)


def val_network(model, criterion, dataloader):
    with torch.no_grad():
        model.eval()
        total_val_loss = 0.
        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.reshape((imgs.size()[0], -1))
            result = model(imgs)
            loss = criterion(result, labels)
            total_val_loss += loss.item()

        return total_val_loss / len(dataloader)


def is_bounded(A):
    ns = linalg.null_space(A)
    if ns.size > 0:
        print('null space is not 0')
        return 1, False
    else:
        A_eq = A.T
        b_eq = np.zeros(A.shape[-1])
        A_ub = -np.eye(A.shape[0])
        b_ub = -np.ones(A.shape[0])
        c = np.ones(A.shape[0])
        output = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        return 0, output.success


def scale_check(model_structure, total_points=10):
    model = relu_network.ReLUNetwork(model_structure)
    points = torch.rand(total_points, model_structure[0])
    total_val = 0
    for point in points:
        point = point
        poly_dict = model.get_polytope(point)
        poly = Polytope.from_polytope_dict(poly_dict, point)
        val, _ = is_bounded(poly.ub_A)
        total_val += val
    print('total val: ', total_val)


def input_size_experiments(model_structure, max_power=5):
    input_sizes = (np.power(2, i) for i in range(1, max_power+1))
    time_ex = list()
    for input_size in input_sizes:
        model = relu_network.ReLUNetwork([input_size] + model_structure)
        point = torch.rand(1, input_size)
        poly_dict = model.get_polytope(point)
        poly = Polytope.from_polytope_dict(poly_dict, point)
        print("current input size:", input_size)
        start = time.time()
        print(poly.ub_A.dtype)
        print(is_bounded(poly.ub_A))
        end = time.time()
        print('------------- execution time:', end-start)
        time_ex.append(end-start)
    return time_ex


def bit_size_experiments(model_structure,  dtypes=None):
    time_ex = list()
    for dtype in dtypes:
        model = relu_network.ReLUNetwork(model_structure)
        point = torch.rand(1, model_structure[0])
        poly_dict = model.get_polytope(point)
        poly = Polytope.from_polytope_dict(poly_dict, point)
        print("current dtype:", dtype)
        A = poly.ub_A.astype(dtype)
        start = time.time()
        print(is_bounded(A))
        end = time.time()
        print('------------- execution time:', end-start)
        time_ex.append(end-start)
    return time_ex

def plot_time(time_total):
    plt.plot(time_total)
    plt.xticks(range(len(time_total)),  [np.power(2, i) for i in range(1, len(time_total)+1)])
    plt.show()

if __name__ == '__main__':
    signal = False
    transforms = Compose([Resize(RESIZE), ToTensor()])
    dataset = get_dataset(PATH_TO_SAVE, transforms, d_type=DATASET_TYPE)
    dataloader_train = DataLoader(dataset, BATCH_SIZE, num_workers=NUM_WORKERS)

    dataset_val = get_dataset(PATH_TO_SAVE, transforms, False, d_type=DATASET_TYPE)
    dataloader_val = DataLoader(dataset_val, BATCH_SIZE, num_workers=NUM_WORKERS)

    model = relu_network.ReLUNetwork([RESIZE*RESIZE] + HIDDEN_SIZE + [OUTPUT_SIZE])
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    if signal:
        for i in range(1, ITERATIONS + 1):
            train_loss = train_network(model, optimizer, criterion, dataloader_train)
            print("epoch: {}\n train loss: {}".format(i, train_loss))
            val_loss = val_network(model, criterion, dataloader_val)
            print(" val loss: {}".format(val_loss))
    else:
       # scale_check([RESIZE] + HIDDEN_SIZE + [2], 1000)

        #for i in [2, 10, 50, 100, 300]:
        #    print('------------ current size:', i)
        #    time_total = bit_size_experiments([i]+HIDDEN_SIZE + [2], [np.float16, np.float32, np.float64, np.float128])
        #    print('--------------------------')
        time_total = input_size_experiments(HIDDEN_SIZE + [2], 15)
        plot_time(time_total)
        #point = torch.rand(1, RESIZE*RESIZE)
        #config = model.input_config(point)
        #polytope_dict = model.get_polytope(point)
        #poly = Polytope.from_polytope_dict(polytope_dict, point)









