import numpy as np
# from sklearn.datasets import make_swiss_roll
import torch

from .base import Dataset, SyntheticDataset
from ..framework.util import spaces




class SwissRollDataset(SyntheticDataset):
	def __init__(self, n_samples=100, noise=0., target_theta=True,
	             Ax=np.pi, Ay=21., Az=np.pi, freq=0.5, tmin=3., tmax=9.,
	             **kwargs):
		super().__init__(**kwargs)

		self.n_samples = n_samples
		self.noise_std = noise

		assert Ax > 0 and Ay > 0 and Az > 0 and freq > 0 and tmax > tmin, f'invalid parameters: ' \
		                                                                  f'{Ax} {Ay} {Az} {freq} {tmax} {tmin}'
		self.Ax, self.Ay, self.Az = Ax, Ay, Az
		self.freq = freq
		self.tmin, self.tmax = tmin, tmax

		self._target_theta = target_theta

		lbl_space = spaces.JointSpace(
			spaces.BoundDim(min=tmin, max=tmax),
			spaces.BoundDim(min=0., max=1.),
		)
		# by default, labels are sampled by generating sampling from the label space

		obs_space = spaces.JointSpace(
			spaces.BoundDim(min=-Ax * tmax, max=Ax * tmax),
			spaces.BoundDim(min=0., max=self.Ay),
			spaces.BoundDim(min=-Az * tmax, max=Az * tmax),
		)

		self.register_buffer('label', space=lbl_space)
		if self._target_theta:
			self.register_buffer('target', space=lbl_space[0])
		self.register_buffer('observation', space=obs_space)


	def _generate_noise(self, N, seed=None, gen=None):
		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen
		return self.noise_std * torch.randn(N, 3, generator=gen)


	def generate_observation_from_mechanism(self, mechanism, seed=None, gen=None):
		theta = mechanism.narrow(-1,0,1)
		height = mechanism.narrow(-1,1,1)

		pts = torch.cat([
			self.Ax * theta * theta.mul(self.freq).cos(),
			self.Ay * height,
			self.Az * theta * theta.mul(self.freq).sin(),
		], -1) + self._generate_noise(len(theta), seed=seed, gen=gen)
		return pts


	def _load(self, *args, **kwargs):
		lbls = self.generate_mechanism(self.n_samples)

		self.buffers['label'].set_data(lbls)
		self.buffers['observation'].set_data(self._generate_observation(lbls))
		if self._target_theta:
			self.buffers['target'].set_data(lbls.narrow(-1,0,1))

		super()._load(*args, **kwargs)






