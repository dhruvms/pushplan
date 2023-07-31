import copy
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from constants import DEVICE
from helpers import shortest_angle_dist

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

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_fn_cvae(recon_x, x, mu, logvar):
	mse = nn.MSELoss()
	recon = mse(recon_x[:, :2], x[:, :2])

	# torch version of shortest_angle_dist
	ang_diff = recon_x[:, 2] - x[:, 2]
	test = torch.abs(ang_diff) > 2*np.pi
	if (torch.any(test)):
		ang_diff[test] = torch.fmod(ang_diff[test], 2*np.pi)
	while (torch.any(ang_diff < -np.pi)):
		test = ang_diff < -np.pi
		ang_diff[test] += 2*np.pi
	while (torch.any(ang_diff > np.pi)):
		test = ang_diff > np.pi
		ang_diff[test] -= 2*np.pi
	recon += torch.mean(torch.abs(ang_diff))

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return recon + kld

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_fn_pose_cvae(recon_x, x, mu, logvar):
	mae = nn.L1Loss()
	mse = nn.MSELoss()

	recon = mae(recon_x[:, 3:], x[:, 3:]) + mse(recon_x[:, :3], x[:, :3])
	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	return recon + kld

def eval_fn_cvae(model, latent_dim, C_eval, X_eval, num_eval=5):
	with torch.no_grad():
		C_eval = torch.repeat_interleave(C_eval, num_eval, dim=0)
		Z_eval = torch.randn(C_eval.shape[0], latent_dim).to(DEVICE)

		# num_eval sample prediction per validation point
		X_eval_pred = model.decode(Z_eval, C_eval).data.cpu().numpy()
		# average every num_eval rows
		X_eval_pred = X_eval_pred.transpose().reshape(-1, num_eval).mean(axis=1).reshape(X_eval.shape[1], -1).transpose()

		ang_diff = shortest_angle_dist(X_eval_pred[:, 2], X_eval[:, 2].cpu().numpy())
		ang_loss = np.fabs(ang_diff)[:, None]

		mse = np.sum((X_eval_pred[:, :2] - X_eval[:, :2].cpu().numpy())**2, axis=1, keepdims=True)
		recon = np.mean(ang_loss + mse)

	return recon

def eval_fn_pose_cvae(model, latent_dim, C_eval, X_eval, num_eval=50):
	with torch.no_grad():
		C_eval = torch.repeat_interleave(C_eval, num_eval, dim=0)
		Z_eval = torch.randn(C_eval.shape[0], latent_dim).to(DEVICE)

		# num_eval sample prediction per validation point
		X_eval_pred = model.decode(Z_eval, C_eval).data.cpu().numpy()
		# average every num_eval rows
		X_eval_pred = X_eval_pred.transpose().reshape(-1, num_eval).mean(axis=1).reshape(X_eval.shape[1] - 3, -1).transpose()

		mae = np.sum(np.abs(X_eval_pred[:, 3:] - X_eval[:, 3:-3].cpu().numpy()), axis=1, keepdims=True)
		mse = np.sum((X_eval_pred[:, :3] - X_eval[:, :3].cpu().numpy())**2, axis=1, keepdims=True)
		recon = np.mean(mae + mse)

	return recon

# Helper function to train one model
def model_train_ik(model, X_train, y_train, X_val, y_val, epochs=1000):
	model = model.to(DEVICE)
	# loss function and optimizer
	loss_fn = nn.BCELoss()  # binary cross entropy
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	N = X_train.shape[0]
	n_epochs = epochs   # number of epochs to run
	batch_size = 1<<(int(np.sqrt(N))-1).bit_length()  # size of each batch
	batch_start = torch.arange(0, len(X_train), batch_size)

	# Hold the best model
	best_acc = -np.inf   # init to negative infinity
	best_weights = None
	best_acc_update_epoch = -1

	for epoch in range(n_epochs):
		idxs = torch.tensor(np.random.permutation(np.arange(0, N)))
		model.train()
		with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=epoch % 50 != 0) as bar:
			bar.set_description("Epoch {}".format(epoch))
			for start in bar:
				batch_idxs = idxs[start:start+batch_size]
				# take a batch
				X_batch = X_train[batch_idxs]
				y_batch = y_train[batch_idxs]

				# forward pass
				y_pred = model(X_batch)
				loss = loss_fn(y_pred, y_batch)

				# backward pass
				optimizer.zero_grad()
				loss.backward()

				# update weights
				optimizer.step()
				# print progress

				acc = (y_pred.round() == y_batch).float().mean()
				bar.set_postfix(
					loss=float(loss),
					acc=float(acc)
				)
		# evaluate accuracy at end of each epoch
		model.eval()
		y_pred = model(X_val)
		acc = (y_pred.round() == y_val).float().mean()
		acc = float(acc)
		if acc > best_acc:
			# print('\tAccuracy improved at epoch {}'.format(epoch))
			best_acc = acc
			best_weights = copy.deepcopy(model.state_dict())
			best_acc_update_epoch = epoch
		elif ((best_acc_update_epoch >= 0) and epoch > best_acc_update_epoch + n_epochs//10):
			print('Training stopped at epoch {}. Last acc update was at epoch {}.'.format(epoch, best_acc_update_epoch))
			break

	# restore model and return best accuracy
	model.load_state_dict(best_weights)
	return best_acc

# Helper function to train one model
def model_train_cvae(model, C_train, X_train, latent_dim, C_eval, X_eval, loss_fn, eval_fn, epochs=1000):
	model = model.to(DEVICE)
	# loss function and optimizer
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	N = X_train.shape[0]
	n_epochs = epochs   # number of epochs to run
	batch_size = 1<<(int(np.sqrt(N))-1).bit_length()  # size of each batch
	batch_start = torch.arange(0, len(X_train), batch_size)

	# Hold the best model
	best_recon = np.inf   # init to negative infinity
	best_weights = None
	best_recon_update_epoch = -1

	for epoch in range(n_epochs):
		idxs = torch.tensor(np.random.permutation(np.arange(0, N)))
		model.train()
		with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=epoch % 50 != 0) as bar:
			bar.set_description("Epoch {}".format(epoch))
			for start in bar:
				batch_idxs = idxs[start:start+batch_size]
				# take a batch
				C_batch = C_train[batch_idxs]
				X_batch = X_train[batch_idxs]

				# forward pass
				recon_batch, mu, logvar = model(X_batch, C_batch)
				loss = loss_fn(recon_batch, X_batch, mu, logvar)

				# backward pass
				optimizer.zero_grad()
				loss.backward()

				# update weights
				optimizer.step()
				# print progress

				# acc = (X_pred.round() == X_batch).float().mean()
				# bar.set_postfix(
				#     loss=float(loss),
				#     acc=float(acc)
				# )

		# evaluate model at end of each epoch
		recon = eval_fn(model, latent_dim, C_eval, X_eval)
		if recon < best_recon:
			best_recon = recon
			best_weights = copy.deepcopy(model.state_dict())
			best_recon_update_epoch = epoch
		elif ((best_recon_update_epoch >= 0) and epoch > best_recon_update_epoch + n_epochs//10):
			print('Training stopped at epoch {}. Last acc update was at epoch {}.'.format(epoch, best_recon_update_epoch))
			break

	# restore model and return best accuracy
	model.load_state_dict(best_weights)
	return best_recon
