import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import revtorch as rv


class ReVAE(nn.Module):
    def __init__(self):
        super(ReVAE, self).__init__()
        self.fc1 = nn.Linear(3072, 400)
        self.fc21 = nn.Linear(400, 30)
        self.fc22 = nn.Linear(400, 30)
        self.fc3 = nn.Linear(30, 400)
        self.fc4 = nn.Linear(400, 3072)

        # f and g must both be a nn.Module whos output has the same shape as its input
        f_func = nn.Sequential(nn.ReLU(), nn.Linear(200, 200))
        g_func = nn.Sequential(nn.ReLU(), nn.Linear(200, 200))

        blocks = [rv.ReversibleBlock(f_func, g_func) for i in range(2)]

        # pack all reversible blocks into a reversible sequence
        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.sequence(h1)
        return self.fc21(h1), self.fc22(h1)

    # key part of VAE
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.sequence(h3)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3*1024))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3072), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
