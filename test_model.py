import pytest
import torch

from run import create_parser
from model import ReVAE, loss_function
from torch import optim
from torchvision import datasets, transforms


@pytest.fixture
def model():
    model = ReVAE()
    return model


@pytest.fixture
def test_data():
    transform = transforms.Compose(
        [transforms.ToTensor()])

    return torch.utils.datasDataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform),
        batch_size=128, shuffle=True)


@pytest.fixture
def train_data():
    transform = transforms.Compose(
        [transforms.ToTensor()])

    return torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transform),
        batch_size=128, shuffle=True)


def test_train(model, train_data):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print('testing')
    epoch = 'pytest'
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_data):

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        assert mu.size()==logvar.size()
        loss = loss_function(recon_batch, data, mu, logvar)
        assert loss != 0
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_data.dataset),
                       100. * batch_idx / len(train_data),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_data.dataset)))





