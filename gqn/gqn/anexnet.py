from .gquerynet import (ResidualConnect, PoolSceneEncoder, PriorFactor, InferenceCoreInlet, 
							RecurrentCore, PosteriorFactor, GeneratorCoreDelta, GeneratorCoreOutlet)
import torch, imageio, os, math, random
import torch.nn as nn
import numpy as np
from torch.nn.functional import interpolate
from torch.distributions import Normal
from numpy.random import randint
from tqdm import tqdm


class RecurrentAnExEncoder(nn.Module):
	def __init__(self, input_channels=512, output_channels=256, num_layers=2):        
		super(RecurrentAnExEncoder, self).__init__()
		self.input_channels = input_channels
		self.output_channels = output_channels

		self.layer1 = nn.LSTM(input_size=input_channels, hidden_size=output_channels, num_layers=num_layers, batch_first=True)

		print(self.__class__.__name__, sum(map(lambda p: p.numel(), self.parameters())))
	def forward(self, inp):
		out, (c, h) = self.layer1(inp)
		return out 


class AnExNet(nn.Module):
	def __init__(self, r=256, v=7, xi=256, z=64, u=128, hg=64, he=64, bias=True):
		"""
		: channel sizes :
			r  - scene encoder output
			v  - viewpoint, default 7, (X, Y, Z sin(P), cos(P), sin(H), cos(H))
			xi - view inlet output
			z  - variational sample
			u  - generator core output
			hg - generator core hidden
			he - inference core hidden
		"""
		super(AnExNet, self).__init__()
		self.bias = bias
		self.channel_sizes = {
			'r': r, 'v': v, 'xi': xi, 'z': z, 'u': u, 'hg': hg, 'he': he
		}

		self.scene_encoder = PoolSceneEncoder(output_channels=r, hidden_channels=64)
		self.icore_inlet = InferenceCoreInlet(output_channels=xi, hidden_channels=64)
		self.anex_encoder = RecurrentAnExEncoder(input_channels=r*2, output_channels=r)
		self.prior_factor = PriorFactor(input_channels=hg, output_channels=z, hidden_channels=[128, 64])
		self.icore_cell = RecurrentCore(input_channels=xi+r+hg+u, hidden_channels=64, kernel_size=(5, 5))
		self.posterior_factor = PosteriorFactor(input_channels=he, output_channels=z, hidden_channels=[128, 64])
		self.gcore_cell = RecurrentCore(input_channels=v+r+z, hidden_channels=64, kernel_size=(5, 5))
		self.gcore_delta = GeneratorCoreDelta(input_channels=hg, output_channels=u)
		self.gcore_outlet = GeneratorCoreOutlet(input_channels=u, hidden_channels=[128, 64])

		f = 1.15
		self.parameter_groups_lr_factors = [
			f ** 5,	# scene_encoder		(5)
			f ** 5,	# icore_inlet		(5)
			f ** 4,	# anex_encoder		(4)
			f ** 3,	# posterior_factor	(3)
			f ** 2,	# icore_cell		(2)
			f ** 3,	# posterior_factor	(3)
			f ** 2,	# gcore_cell		(2)
			f ** 1,	# gcore_delta		(1)
			f ** 0	# gcore_outlet		(0)
			]

		print(self.__class__.__name__, sum(map(lambda p: p.numel(), self.parameters())))

	def parameter_groups(self, lr=5e-5):
		pgrp = [
			{'params': self.scene_encoder.parameters(),    'lr': lr},
			{'params': self.icore_inlet.parameters(),      'lr': lr},
			{'params': self.anex_encoder.parameters(),     'lr': lr},
			{'params': self.prior_factor.parameters(),     'lr': lr},
			{'params': self.icore_cell.parameters(),       'lr': lr},
			{'params': self.posterior_factor.parameters(), 'lr': lr},
			{'params': self.gcore_cell.parameters(),       'lr': lr},
			{'params': self.gcore_delta.parameters(),      'lr': lr},
			{'params': self.gcore_outlet.parameters(),     'lr': lr}]

		return pgrp

	def _xr_st(self, BS, K, T, xsk, vsk, xtk, vtk):
		rsize = self.channel_sizes['r']

		# composition of static scenes
		xsr  = self.scene_encoder(xsk, vsk)
		xsr = xsr.view(BS, K, rsize)
		xsr = xsr.sum(1, keepdim=True)

		# composition of time-dependent scenes
		xtr = self.scene_encoder(xtk, vtk)
		xtr = xtr.view(BS, T, rsize)
		fr0 = xtr[:, 0:1]
		xtr = self.anex_encoder(torch.cat([xtr, xsr.expand(-1, T, -1)], dim=2))

		# add first frame to xsr
		xsr = xsr + fr0

		return xsr, xtr


	def forward(self, K, T, xsk, vsk, xtk, vtk, xq, vq, ar=16):
		"""
			K: number of scenes
			T: number of time steps in playback
			xsk: static scenes [BS * K, channel, height, width]
			vsk: viewpoints corresponding to each static scene
			xtk: dynamic scenes
			vtk: 
			vq: [BS * (T+1), 7, 1, 1]; vsq, vtq (replicate T times)
			
		"""
		BS = int(xsk.size(0) / K)
		DEV = xsk.device

		# sufficient statistics, mean (loc) and log-variance (lva),
		# for generative densities (pd) and recognition densities (qd)
		pd_loc = []
		pd_lva = []
		qd_loc = []
		qd_lva = []

		rsize = self.channel_sizes['r']
		hgt = wdt = 1  # TODO: this is for 64x64 inputs only

		xsr, xtr = slef._xr_st(BS, K, T, xsk, vsk, xtk, vtk)

		# hidden and cell states for recurrent layers
		hiddg = torch.zeros(BS * (T + 1), self.channel_sizes['hg'], 16*hgt, 16*wdt, device=DEV)
		cellg = torch.zeros(BS * (T + 1), self.channel_sizes['hg'], 16*hgt, 16*wdt, device=DEV)
		hiddi = torch.zeros(BS * (T + 1), self.channel_sizes['he'], 16*hgt, 16*wdt, device=DEV)
		celli = torch.zeros(BS * (T + 1), self.channel_sizes['he'], 16*hgt, 16*wdt, device=DEV)
		u     = torch.zeros(BS * (T + 1), self.channel_sizes['u'],  16*hgt, 16*wdt, device=DEV)

		repq = self.icore_inlet(xq, vq)
		repk = torch.cat([xsr, xtr], dim=1).view(BS * (T + 1), rsize, hgt, wdt)
		repk = interpolate(repk, scale_factor=repq.size(2)/repk.size(2), mode='nearest')

		# recurrent sampling
		for _ in range(ar):
			_, pdloc, pdlva = self.prior_factor(hiddg)

			hiddi, celli = self.icore_cell(torch.cat([repq, repk, hiddg, u], dim=1), hiddi, celli)

			qd, qdloc, qdlva = self.posterior_factor(hiddi)
			z = qd.rsample()

			inp = torch.cat([vq.expand(-1, -1, z.size(2), z.size(2)), repk, z], dim=1)
			hiddg, cellg = self.gcore_cell(inp, hiddg, cellg)

			u = u + self.gcore_delta(hiddg)

			pd_loc.append(pdloc)
			pd_lva.append(pdlva)
			qd_loc.append(qdloc)
			qd_lva.append(qdlva)

		x = self.gcore_outlet(u)
		return x, [qd_loc, qd_lva, pd_loc, pd_lva]

	def predict(self, K, T, xsk, vsk, xtk, vtk, vq, ar=16): 
		BS = int(xsk.size(0) / K)
		DEV = xsk.device

		rsize = self.channel_sizes['r']
		hgt = wdt = 1  # TODO: this is for 64x64 inputs only

		xsr, xtr = self._xr_st(BS, K, T, xsk, vsk, xtk, vtk)
		
		hiddg = torch.zeros(BS * (T + 1), self.channel_sizes['hg'], 16*hgt, 16*wdt, device=DEV)
		cellg = torch.zeros(BS * (T + 1), self.channel_sizes['hg'], 16*hgt, 16*wdt, device=DEV)
		u     = torch.zeros(BS * (T + 1), self.channel_sizes['u'],  16*hgt, 16*wdt, device=DEV)

		repk = torch.cat([xsr, xtr], dim=1).view(BS * (T + 1), rsize, hgt, wdt)
		repk = interpolate(repk, scale_factor=u.size(2)/repk.size(2), mode='nearest')

		for _ in range(ar):
			pd, pdloc, pdlva = self.prior_factor(hiddg)
			z = pd.sample()
			inp = torch.cat([vq.expand(-1, -1, z.size(2), z.size(2)), repk, z], dim=1)
			hiddg, cellg = self.gcore_cell(inp, hiddg, cellg)
			u = u + self.gcore_delta(hiddg)

		x = self.gcore_outlet(u)

		return x

	def lock_r(self, K, T, xsk, vsk, xtk, vtk):
		BS = int(xsk.size(0) / K)
		rsize = self.channel_sizes['r']
		hgt = wdt = 1

		xsr, xtr = self._xr_st(BS, K, T, xsk, vsk, xtk, vtk)

		repk = torch.cat([xsr, xtr], dim=1).view(BS * (T + 1), rsize, hgt, wdt)
		repk = interpolate(repk, scale_factor=16/repk.size(2), mode='nearest')

		self.locked_r = repk

	def predict_r(self, vq, idx, ar=16):
		BS = len(idx)
		DEV = vq.device
		hgt = wdt = 1

		repk = self.locked_r[idx]
		hiddg = torch.zeros(BS, self.channel_sizes['hg'], 16*hgt, 16*wdt, device=DEV)
		cellg = torch.zeros(BS, self.channel_sizes['hg'], 16*hgt, 16*wdt, device=DEV)
		u     = torch.zeros(BS, self.channel_sizes['u'],  16*hgt, 16*wdt, device=DEV)

		for _ in range(ar):
			pd, pdloc, pdlva = self.prior_factor(hiddg)
			z = pd.sample()
			inp = torch.cat([vq.expand(-1, -1, z.size(2), z.size(2)), repk, z], dim=1)
			hiddg, cellg = self.gcore_cell(inp, hiddg, cellg)
			u = u + self.gcore_delta(hiddg)

		x = self.gcore_outlet(u)

		return x


def sample_data(base_path, B, K, T, N, Kmax=20, Tmax=24, cuda=True):
	# B -- batch size
	# K -- number of contextual views  
	# T -- number of time steps in playback
	# N -- scnen number [from, to]
	# Kmax -- maximum number of views avaiable
	# Tmax -- maximum playback length
	xsk = []
	vsk = []
	xtk = []
	vtk = []
	xsq = []
	vsq = []
	xtq = []
	vtq = []

	if type(N) is int:
		N = [0, N]
	else:
		assert len(N) == 2

	batch_indices = randint(N[0], N[1], B)
	for b in batch_indices:
		scene_path = os.path.join(base_path, '{:08d}').format(b)

		# collect contextual views
		npyfile = os.path.join(scene_path, 'static-xyzhp.npy')
		xyzph = np.load(npyfile)
		all_k = np.random.choice(Kmax, K + 1, replace=False)
		for k in all_k[:-1]:
			imgfile = os.path.join(scene_path, '{:08d}_{:02d}.jpg').format(b, k)
			img = imageio.imread(imgfile).transpose(2, 0, 1) / 255
			xsk.append(img)
			vsk.append(xyzph[k])

		k = all_k[-1]
		imgfile = os.path.join(scene_path, '{:08d}_{:02d}.jpg').format(b, k)
		img = imageio.imread(imgfile).transpose(2, 0, 1) / 255
		xsq.append(img)
		vsq.append(xyzph[k])

		# collect contextual playback
		xyzph0 = np.load(os.path.join(scene_path, 'motion0-xyzhp.npy'))
		xyzph1 = np.load(os.path.join(scene_path, 'motion1-xyzhp.npy'))
		t0 = randint(0, Tmax - T + 1)
		for t in range(t0, t0 + T):
			imgfile = os.path.join(scene_path, 'motion0-{:08d}_{:02d}.jpg').format(b, t)
			img = imageio.imread(imgfile).transpose(2, 0, 1) / 255
			xtk.append(img)
			vtk.append(xyzph0[t])

			imgfile = os.path.join(scene_path, 'motion1-{:08d}_{:02d}.jpg').format(b, t)
			img = imageio.imread(imgfile).transpose(2, 0, 1) / 255
			xtq.append(img)
			vtq.append(xyzph1[t])
	
	xsk = torch.tensor(xsk, dtype=torch.float32, device='cuda' if cuda else 'cpu')
	vsk = torch.tensor(vsk, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B * K, 7, 1, 1)
	xtk = torch.tensor(xtk, dtype=torch.float32, device='cuda' if cuda else 'cpu')
	vtk = torch.tensor(vtk, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B * T, 7, 1, 1)
	xsq = torch.tensor(xsq, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B, 1, 3, 64, 64)
	vsq = torch.tensor(vsq, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B, 1, 7, 1, 1)
	xtq = torch.tensor(xtq, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B, T, 3, 64, 64)
	vtq = torch.tensor(vtq, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(B, T, 7, 1, 1)

	xq = torch.cat([xsq, xtq], dim=1).view(B * (T + 1), 3, 64, 64)
	vq = torch.cat([vsq, vtq], dim=1).view(B * (T + 1), 7,  1,  1)
	
	return K, T, xsk, vsk, xtk, vtk, xq, vq, batch_indices


def kl_divergence(qm, qv, pm, pv):
	return (-pv + qv).exp().sum(1) + (qm - pm).pow(2).div(pv.exp()).sum(1) + pv.sum(1) - qv.sum(1)


def anneal_sigma(epoch, epoch_max, sigma_min=0.7, sigma_max=2.0):
	return max(sigma_min + (sigma_max - sigma_min) * (1 - epoch / epoch_max), sigma_min)


def anneal_lr(optimiser, epoch, factors, n=1.6e6, lr_min=5e-5, lr_max=5e-4):
	lr = max(lr_min + (lr_max - lr_min) * (1 - epoch / n), lr_min)
	for param_group, factor in zip(optimiser.param_groups, factors):
		param_group['lr'] = lr * factor
		

def mock_train(model, optimiser, data_base_path, 
			   train_scene_range,
			   epochs=2e6, batches=32, K=15, Kmax=20, ar=16, 
			   sigma_min=0.7, sigma_max=2.0, 
			   lr_max=5e-4, lr_min=1e-4, 
			   use_kl=True,
			   cuda=True,
			   fhndl=None, save_path='',
			   step_every=1, save_every=10000, from_epoch=0):
	"""mock_train: training without testing."""

	if fhndl is None:
		print('** models will not be saved.')
	else:
		assert os.path.exists(save_path), print('** save path `{}` does not exist.'.format(save_path))

	for epoch in tqdm(range(epochs)):
		if epoch < from_epoch:
			continue

		T = random.randint(1, 24)
		k, t, xsk, vsk, xtk, vtk, xq, vq = sample_data(data_base_path, batches, randint(1, K), T, train_scene_range, Kmax=Kmax, Tmax=24, cuda=cuda)
		x, qpstat = model(k, t, xsk, vsk, xtk, vtk, xq, vq, ar=ar)

		sigma = anneal_sigma(epoch, epochs, sigma_min, sigma_max)
		anneal_lr(optimiser, epoch, model.parameter_groups_lr_factors, .8 * epochs, lr_min, lr_max)

		kl = 0
		if use_kl:
			for m0, v0, m1, v1 in zip(*qpstat):
				kl = kl + kl_divergence(m0, v0, m1, v1).sum(2).sum(1)
			kl = kl.mean()
			
		sqe_train = (x - xq).pow(2).sum(3).sum(2).sum(1).mean()
		
		(1./sigma * sqe_train + kl).backward()
		if (epoch + 1) % step_every == 0:
			optimiser.step()
			optimiser.zero_grad()

		if fhndl is not None:
			np.savetxt(fhndl, [[float(sqe_train)]])
			fhndl.flush()

			if epoch == 0 or (epoch + 1) % save_every == 0:
				torch.save(model.state_dict(), os.path.join(save_path, 'model_{:08d}.pth'.format(epoch)))


def gauss_kernel(size=5, sigma=1.0, channels=1, cuda=True):
	grid = np.mgrid[0:size,0:size].T
	gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
	kernel = np.sum(gaussian(grid), axis=2)
	kernel /= np.sum(kernel)
	kernel = torch.tensor(kernel, dtype=torch.float32, device='cuda' if cuda else 'cpu').view(1, 1, size, size)
	return kernel.expand(channels, channels, -1, -1)


def conv_gauss(inp, stride=1, k_size=5, sigma=1.6, channels=1, cuda=True):
	kernel = gauss_kernel(size=k_size, sigma=sigma, channels=channels, cuda=cuda)
	result = nn.functional.conv2d(inp, kernel, stride=stride, padding=k_size//2)
	return result


def laplacian_pyramid(inp, levels, channels=1, cuda=True):
	pyramid = []
	curr = inp
	for level in range(levels):
		smooth = conv_gauss(curr, stride=1, k_size=5, sigma=2.0, channels=channels, cuda=cuda)
		diff = curr - smooth
		pyramid.append(diff)
		curr = nn.functional.avg_pool2d(smooth, (2, 2), stride=(2, 2), padding=0)
		pyramid.append(curr)
	return pyramid


def laploss(inp1, inp2, levels=3, channels=1, p=1, cuda=True):
	bs = inp1.size(0)
	pyr1 = laplacian_pyramid(inp1, levels, channels=channels, cuda=cuda)
	pyr2 = laplacian_pyramid(inp2, levels, channels=channels, cuda=cuda)
	if p == 1:
		loss = [math.pow(2, -2 * j) * torch.norm(a - b, p=1) / float(a.numel()) for j, (a, b) in enumerate(zip(pyr1, pyr2))]
	elif p == 2:
		loss = [math.pow(2, -2 * j) * (a - b).pow(2).sum(3).sum(2).sum(1).mean() for j, (a, b) in enumerate(zip(pyr1, pyr2))]

	loss = sum(loss) * bs
	return loss


def mock_train_laploss(model, optimiser, data_base_path, 
					   train_scene_range,
					   epochs=2e6, batches=32, K=15, Kmax=20, ar=16, 
					   sigma_min=0.7, sigma_max=2.0, 
					   lr_max=5e-4, lr_min=1e-4, 
					   use_kl=True, 
					   cuda=True,
					   fhndl=None, save_path='',
					   step_every=1, save_every=10000, from_epoch=0):
	"""mock_train: training without testing."""

	if fhndl is None:
		print('** models will not be saved.')
	else:
		assert os.path.exists(save_path), print('** save path `{}` does not exist.'.format(save_path))

	for epoch in tqdm(range(epochs)):
		if epoch < from_epoch:
			continue

		T = random.randint(1, 24)
		k, t, xsk, vsk, xtk, vtk, xq, vq = sample_data(data_base_path, batches, randint(1, K), T, train_scene_range, Kmax=Kmax, Tmax=24, cuda=cuda)
		x, qpstat = model(k, t, xsk, vsk, xtk, vtk, xq, vq, ar=ar)

		sigma = anneal_sigma(epoch, epochs, sigma_min, sigma_max)
		anneal_lr(optimiser, epoch, model.parameter_groups_lr_factors, .8 * epochs, lr_min, lr_max)

		kl = 0
		if use_kl:
			for m0, v0, m1, v1 in zip(*qpstat):
				kl = kl + kl_divergence(m0, v0, m1, v1).sum(2).sum(1)
			kl = kl.mean()
		
		ll_train = laploss(x, xq, levels=3, channels=3, p=2, cuda=cuda)
		
		(1./sigma * ll_train + kl).backward()
		if (epoch + 1) % step_every == 0:
			optimiser.step()
			optimiser.zero_grad()

		if fhndl is not None:
			np.savetxt(fhndl, [[float(ll_train)]])
			fhndl.flush()

			if epoch == 0 or (epoch + 1) % save_every == 0:
				torch.save(model.state_dict(), os.path.join(save_path, 'model_{:08d}.pth'.format(epoch)))