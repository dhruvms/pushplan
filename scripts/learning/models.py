import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

activations = nn.ModuleDict([
				['relu', nn.ReLU()],
				['lrelu', nn.LeakyReLU()],
				['selu', nn.SELU()],
				['tanh', nn.Tanh()],
			])

def one_layer(in_f, out_f, activation='relu', *args, **kwargs):

	return nn.Sequential(
		nn.Linear(in_f, out_f, *args, **kwargs),
		activations[activation]
	)

# Define two models
class BCNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.initialised = False

	def initialise(self, in_dim, out_dim, activation='relu', layers=None, h_sizes=None, *args, **kwargs):
		if self.initialised:
			return

		if layers is not None:
			assert h_sizes is not None
			assert len(h_sizes) == layers-1
			self.h_sizes = [in_dim] + copy.deepcopy(h_sizes)

			self.layers = nn.Sequential(*[one_layer(self.h_sizes[i-1], self.h_sizes[i], activation=activation, *args, **kwargs) for i in range(1, layers)])
			self.output = nn.Linear(self.h_sizes[-1], out_dim, *args, **kwargs)
		else:
			self.layers = one_layer(in_dim, in_dim, activation)
			self.output = nn.Linear(in_dim, out_dim)
		self.sigmoid = nn.Sigmoid()

		self.initialised = True

	def forward(self, x):
		return self.sigmoid(self.output(self.layers(x)))

class CVAE(nn.Module):
	def __init__(self):
		super().__init__()
		self.initialised = False

	def initialise(self, in_dim, cond_dim, latent_dim, activation='relu', layers=None, h_sizes=None, *args, **kwargs):
		if self.initialised:
			return

		self.in_dim = in_dim
		self.cond_dim = cond_dim
		self.latent_dim = latent_dim
		self.layers = layers
		self.h_sizes = copy.deepcopy(h_sizes)

		# encode
		self.h_sizes.insert(0, in_dim + cond_dim)
		self.encoder = nn.Sequential(*[one_layer(self.h_sizes[i-1], self.h_sizes[i], activation=activation, *args, **kwargs) for i in range(1, self.layers)])
		self.z_mu = nn.Linear(self.h_sizes[-1], latent_dim, *args, **kwargs)
		self.z_logvar = nn.Linear(self.h_sizes[-1], latent_dim, *args, **kwargs)

		# decode
		self.h_sizes.pop(0)
		self.h_sizes.reverse()
		self.h_sizes.insert(0, latent_dim + cond_dim)
		self.decoder = nn.Sequential(*[one_layer(self.h_sizes[i-1], self.h_sizes[i], activation=activation, *args, **kwargs) for i in range(1, self.layers)])
		self.x_mu = nn.Linear(self.h_sizes[-1], in_dim, *args, **kwargs)
		self.x_logvar = nn.Linear(self.h_sizes[-1], in_dim, *args, **kwargs)

		self.initialised = True

	def encode(self, x, c): # Q(z|x, c)
		'''
		x: (bs, in_dim)
		c: (bs, cond_dim)
		'''
		inputs = torch.cat([x, c], 1) # (bs, in_dim + cond_dim)
		h_enc = self.encoder(inputs)
		z_mu = self.z_mu(h_enc)
		z_logvar = self.z_logvar(h_enc)
		return z_mu, z_logvar

	def reparameterize_iso(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def output_dist(self, z, c): # P(x|z, c)
		'''
		z: (bs, latent_dim)
		c: (bs, cond_dim)
		'''
		inputs = torch.cat([z, c], 1) # (bs, latent_dim + cond_dim)
		h_dec = self.decoder(inputs)
		x_mu = self.x_mu(h_dec)
		x_logvar = self.x_logvar(h_dec)
		return x_mu, x_logvar

	def decode(self, z, c): # P(x|z, c)
		'''
		z: (bs, latent_dim)
		c: (bs, cond_dim)
		'''
		mu, logvar = self.output_dist(z, c)
		D = self.reparameterize_iso(mu, logvar)
		return D

	def forward(self, x, c):
		'''
		x: (bs, in_dim)
		c: (bs, cond_dim)
		'''
		mu, logvar = self.encode(x, c)
		z = self.reparameterize_iso(mu, logvar)
		return self.decode(z, c), mu, logvar

class PoseCVAE(nn.Module):
	def __init__(self):
		super().__init__()
		self.initialised = False
		self.cvae = CVAE()

	def initialise(self, in_dim, cond_dim, latent_dim, activation='relu', layers=None, h_sizes=None, *args, **kwargs):
		if self.initialised:
			return

		self.cvae.initialise(in_dim, cond_dim, latent_dim, activation=activation, layers=layers, h_sizes=h_sizes, *args, **kwargs)

	def decode(self, z, c): # P(x|z, c)
		'''
		z: (bs, latent_dim)
		c: (bs, cond_dim)
		'''
		D = self.cvae.decode(z, c)

		# Gram-Schmidt orthonormalisation
		c1 = F.normalize(D[:, -6:-3], p=2, dim=1)
		c2 = F.normalize(D[:, -3:] - (torch.sum(c1 * D[:, -3:], dim=-1).unsqueeze(-1) * c1), p=2, dim=1)
		D = torch.cat([D[:, :-6], c1, c2], 1)

		return D

	def get_rotation_matrix(self, rot6d):
		c1 = rot6d[:, :3]
		c2 = rot6d[:, 3:]
		c3 = torch.cross(c1, c2, dim=1)
		return torch.stack([c1, c2, c3], dim=2)

	def predict_rotation_matrix(self, x, c):
		'''
		x: (bs, in_dim)
		c: (bs, cond_dim)
		'''
		D, mu, logvar = self.forward(x, c)
		R = self.get_rotation_matrix(D[:, -6:])
		return R

	def forward(self, x, c):
		'''
		x: (bs, in_dim)
		c: (bs, cond_dim)
		'''
		D, mu, logvar = self.cvae(x, c)

		# Gram-Schmidt orthonormalisation
		c1 = F.normalize(D[:, -6:-3], p=2, dim=1)
		c2 = F.normalize(D[:, -3:] - (torch.sum(c1 * D[:, -3:], dim=-1).unsqueeze(-1) * c1), p=2, dim=1)
		D = torch.cat([D[:, :-6], c1, c2], 1)

		# recon loss for D should be mean absolute error
		# KL loss for mu, logvar should be as usual
		return D, mu, logvar
