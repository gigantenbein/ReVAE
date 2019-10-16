import logging
import time
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pytest
from guppy import hpy

from model import ReVAE, loss_function

def create_parser():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    return parser, device, args


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        h = hpy()
        logging.info('Memory consumption in bytes: {}'.format(h.heap().size))
        data = data.to(device)

        optimizer.zero_grad() # prevent gradient from accumulating

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()

        train_loss += loss.item()

        loss_ = nn.MSELoss(reduction='mean')

        loss_ = loss_(recon_batch.view(-1, 3072), data.view(-1, 3072))

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_))
            with torch.no_grad():
                sample_ = torch.randn(64, 10).to(device)
                sample_ = model.decode(sample_).cpu()
                save_image(sample_.view(64, 3, 32, 32),
                           'intermediates/sample_' + timestring + str(batch_idx) + '.png')

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


@pytest.mark.skip(reason="name collision of test functions and test of model")
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 3, 32, 32)[:n]])
                save_image(comparison.cpu(),
                         'results_cifar/reconstruction_' + timestring + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    parser, device, args = create_parser()

    transform = transforms.Compose(
        [transforms.ToTensor()])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    cifar_dataset = datasets.CIFAR10('./data', train=True, download=True,
                         transform=transform)

    train_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        cifar_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    logging.basicConfig(filename='memory.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Starting')

    timestamp = time.localtime()
    timestring = "_{}_{}_{}_{}_".format(timestamp.tm_year, timestamp.tm_mon, timestamp.tm_mday, timestamp.tm_hour)

    model = ReVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 10).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, 32, 32),
                       'results_cifar/sample_' + timestring + str(epoch) + '.png')