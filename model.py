import torch
import torch.nn as nn
from torch.nn import functional as F
import revtorch as rv


class Sequence(nn.Module):
    def __init__(self, n_layers):
        super(Sequence, self).__init__()
        # f and g must both be a nn.Module whos output has the same shape as its input
        f_func_enc = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1))
        g_func_enc = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1))

        blocks_enc = [rv.ReversibleBlock(f_func_enc, g_func_enc) for i in range(n_layers)]
        self.sequence_enc = rv.ReversibleSequence(nn.ModuleList(blocks_enc))

    def forward(self, x):
        return self.sequence_enc(x)


class ConvolutionalVAE(nn.Module):
    def __init__(self, n_layers_enc=1, n_layers_dec=1):
        super(ConvolutionalVAE, self).__init__()

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Sequence(3),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)
        )
        self.mu_fc = nn.Linear(4096, 128)
        self.logvar_fc = nn.Linear(4096, 128)

        self.latent_fc = nn.Linear(128, 8 * 8 * 1024)
        self.decoder_layers = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            Sequence(3),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=1, padding=2),
        )

    def encode(self, x):
        h1 = self.encoder_layers(x)
        h1 = h1.view(-1, 4096)
        return self.mu_fc(h1), self.logvar_fc(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.latent_fc(z)
        h3 = self.decoder_layers(h3.view(-1, 1024, 8, 8))
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z).view(-1, 3072), mu, logvar

class ReVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_layers_enc=4, n_layers_dec=4, latent_dim=10):
        super(ReVAE, self).__init__()

        # =============================
        self.fc1 = nn.Linear(3072, 400)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 3072)
        # =============================

        self.bn32 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(3)
        self.bn16 = nn.BatchNorm2d(16)

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
        h1 = F.relu(self.bn32(self.conv1(x)))

        before_size = h1.size()
        h1 = F.relu(self.bn32(self.sequence_enc(h1)))
        after_size = h1.size()

        assert before_size == after_size, 'Size {} is not equal to {}'.format(before_size, after_size)
        h1 = F.relu(self.bn3(self.conv2(h1)))

        h1 = h1.view(-1, 2700)
        return self.fc21(h1), self.fc22(h1)

    def _encode(self, x):
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
        h3 = F.relu(self.bn16(self.conv3(h3)))
        h3 = F.relu(self.bn16(self.conv4(h3)))
        h3 = self.conv5(h3)
        return torch.sigmoid(h3)

    def _decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    assert mu.size() == logvar.size()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3072), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
