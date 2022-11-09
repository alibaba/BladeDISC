# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.testing import assert_allclose
from tests.disc.testing_base import skipTorchNE
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch_blade
import pytest

if torch_blade._is_ltc_available:
    import torch._lazy as ltc
    torch_blade.init_ltc_disc_backend()

import unittest

LOG_INTERVAL=100

## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)


def run_train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        ltc.mark_step()
        train_loss += loss.item() * len(data)
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    print('Train Epoch:{} Average loss: {:.4f}'.format(epoch, train_loss))


def run_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum(dtype=torch.int).item()
            ltc.mark_step()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct * 1.0 / len(test_loader.dataset)

@skipTorchNE("1.12.0")
class TestMnist(unittest.TestCase):
    def mnit(self, device):
        torch.manual_seed(2)
        epochs = 2
        lr = 1.0
        gamma = 0.7
        train_kwargs = {'batch_size': 64}
        test_kwargs = {'batch_size': 1000}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('~/.cache/data', train=True, download=True,
                       transform=transform)
        dataset2 = datasets.MNIST('~/.cache/data', train=False,
                       transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        test_acc = None
        for epoch in range(1, epochs + 1):
            run_train(model, device, train_loader, optimizer, epoch)
            test_acc = run_test(model, device, test_loader)
            scheduler.step()

        return test_acc

    def test_acc(self):
        lazy_device = torch.device('lazy')
        device = torch.device('cpu')
        actual_acc = self.mnit(lazy_device)
        # test on CUDA device
        expect_acc = self.mnit(device)
        print(expect_acc)
        print(actual_acc)
        assert_allclose(expect_acc, actual_acc, rtol=1e-2, atol=0)

if __name__ == '__main__':
    unittest.main()
