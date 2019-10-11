import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import revtorch as rv


class ReVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_layers_enc=1, n_layers_dec=1, latent_dim=10):
        super(ReVAE, self).__init__()

        # =============================
        self.fc1 = nn.Linear(3072, 400)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 3072)
        # =============================

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 3)  # go to 32 channels such that reversible blocks can split it.

        # f and g must both be a nn.Module whos output has the same shape as its input
        f_func_enc = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1))
        g_func_enc = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1))

        blocks_enc = [rv.ReversibleBlock(f_func_enc, g_func_enc) for i in range(n_layers_enc)]

        self.conv2 = nn.Conv2d(32, 3, 3, padding=1)

        self.sequence_enc = rv.ReversibleSequence(nn.ModuleList(blocks_enc))

        self.fc21 = nn.Linear(2700, latent_dim)
        self.fc22 = nn.Linear(2700, latent_dim)

        # Decoder
        self.lin3 = nn.Linear(latent_dim, 400)
        self.lin4 = nn.Linear(400, 8*8*32)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=(2, 2))
        self.conv4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=(2, 2))
        self.conv5 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=(1, 1), padding=1) # padding dimensions

        blocks_dec = [rv.ReversibleBlock(f_func_enc, g_func_enc) for i in range(n_layers_dec)]

        # pack all reversible blocks into a reversible sequence
        self.sequence_dec = rv.ReversibleSequence(nn.ModuleList(blocks_dec))

        self.last = nn.Conv2d(32, 3, 3, padding=1)

    def encode(self, x):
        h1 = F.relu(self.conv1(x))

        before_size = h1.size()
        h1 = F.relu(self.sequence_enc(h1))
        after_size = h1.size()

        assert before_size == after_size, 'Size {} is not equal to {}'.format(before_size, after_size)
        h1 = F.relu(self.conv2(h1))

        h1 = h1.view(-1, 2700)
        return self.fc21(h1), self.fc22(h1)

    def encode_(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.lin3(z))
        h3 = F.relu(self.lin4(h3))
        h3 = h3.view(-1, 32, 8, 8)
        h3 = F.relu(self.conv3(h3))
        h3 = F.relu(self.conv4(h3))
        h3 = self.conv5(h3)
        #h3 = F.relu(self.sequence_dec(h3))
        return torch.sigmoid(h3)

    def _decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        # mu, logvar = self.encode(x.view(-1, 3072))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
# WATCH OUT: probably dimension makes a difference
def loss_function(recon_x, x, mu, logvar):
    assert mu.size() == logvar.size()
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    loss = nn.MSELoss(reduction='mean')

    #BCE = loss(recon_x, x.view(-1, 3072))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
