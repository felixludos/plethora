# from pathlib import Path
# import sys, os
import numpy as np
import torch
# from torch import nn
from torch.distributions import Normal
# from torch.nn import functional as F
#
# from torchvision import datasets, transforms
# import time

from omnibelt import get_printer
# import omnifig as fig



from omnibelt import get_printer, agnosticmethod
from ...framework import hparam, inherit_hparams, models
from .lossless import LosslessCompressionTask, LosslessCompressor

from .task import AbstractLosslessCompressionTask
# from .bits_back import BitsBackCompressor


# The code in these files can be found at https://github.com/bits-back/bits-back
from ...community.bits_back import rans
from ...community.bits_back import util
# from ...community.bits_back.torch_vae.tvae_beta_binomial import BetaBinomialVAE
from ...community.bits_back.torch_vae import tvae_utils
# from ...community.bits_back.torch_vae.torch_mnist_compress import run_compress
# from ...community.bits_back.torch_vae.torch_bin_mnist_compress import run_bin_compress


prt = get_printer(__file__)



class BitsBackCompressionTask(LosslessCompressionTask):

	encoder = hparam(module=models.Encoder)
	decoder = hparam(module=models.Decoder)

	beta_confidence = hparam(1000)
	default_scale = hparam(0.1)


	@agnosticmethod
	def encode(self, observation):
		return self.encoder.encode(observation)


	@agnosticmethod
	def decode(self, latent):
		return self.decoder.decode(latent)


	@hparam(cache=True)
	def compressor(self):
		return BitsBackCompressor(self.encode, self.decode,
		                          obs_shape=self.encoder.din.shape, latent_shape=self.decoder.din.shape,
		                          encoder_device=self.encoder.device, decoder_device=self.decoder.device,
		                          beta_confidence=self.beta_confidence, default_scale=self.default_scale)



class BitsBackCompressor(LosslessCompressor):
	def __init__(self, encode_fn, decode_fn, seed=None, obs_shape=None, latent_shape=None,
	             encoder_in_shape=None, decoder_in_shape=None, as_bytes=False,
	             device=None, encoder_device=None, decoder_device=None,
	             output_distribution='beta', beta_confidence=1000, default_scale=0.1,
	             obs_precision=14, q_precision=14, prior_precision=8, **kwargs):
		super().__init__(**kwargs)
		# if latent_shape is None:
		# 	latent_shape = getattr(encoder, 'latent_dim', encoder.dout)
		# 	if isinstance(latent_shape, int):
		# 		latent_shape = 1, latent_shape
		if isinstance(obs_shape, int):
			obs_shape = obs_shape,
		if isinstance(latent_shape, int):
			latent_shape = latent_shape,
		if encoder_in_shape is None:
			encoder_in_shape = obs_shape
		if decoder_in_shape is None:
			decoder_in_shape = latent_shape
		self._as_bytes = as_bytes
		self.enc_in_shape, self.dec_in_shape = encoder_in_shape, decoder_in_shape
		if len(latent_shape) == 1:
			latent_shape = (1, *latent_shape)
		latent_dim = np.product(latent_shape).item()

		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		if encoder_device is None:
			encoder_device = device
		if decoder_device is None:
			decoder_device = device
		self.enc_device, self.dec_device = encoder_device, decoder_device

		# self.state = None
		self.start_seed_len = latent_dim
		self.rng = np.random.RandomState(seed)

		self.encode = encode_fn
		self.decode = decode_fn

		rec_net = tvae_utils.torch_fun_to_numpy_fun(self._wrapped_encode)
		gen_net = tvae_utils.torch_fun_to_numpy_fun(self._wrapped_decode)

		if output_distribution == 'beta':
			obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
			obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)
		else:
			obs_append = tvae_utils.bernoulli_obs_append(obs_precision)
			obs_pop = tvae_utils.bernoulli_obs_pop(obs_precision)
		self._output_distribution = output_distribution
		# self._beta_variance = 1 - beta_confidence
		assert beta_confidence > 1
		self._beta_confidence = beta_confidence

		self._default_scale = default_scale

		self.vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
		                             prior_precision, q_precision)
		self.vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
		                       prior_precision, q_precision)


	def _wrapped_encode(self, x):
		x = x.to(self.enc_device).div(255).view(-1, *self.enc_in_shape)
		# x = x.to(self.encoder.get_device()).div(255).view(-1, *self.encoder.din)#.unsqueeze(0)
		with torch.no_grad():
			# z = self.encoder.encode(x)
			z = self.encode(x)
		if isinstance(z, Normal):
			return z.loc.cpu(), z.scale.cpu()
		mu = z.cpu()
		return mu, torch.ones_like(mu)*self._default_scale


	def _wrapped_decode(self, z):
		z = z.to(self.dec_device).unsqueeze(0)
		# z = z.to(self.decoder.get_device()).unsqueeze(0)
		with torch.no_grad():
			# x = self.decoder.decode(z)
			x = self.decode(z)
		x = x.cpu().view(z.size(0), -1)
		if self._output_distribution == 'beta':
			x = self._compute_beta_params(x)
		return x


	def _compute_beta_params(self, x):
		# x = x.clamp(min=1e-8, max=1-1e-8)
		a = x.mul(self._beta_confidence).clamp(min=1e-8)
		b = (1-x).mul(self._beta_confidence).clamp(min=1e-8)
		return a,b


	def generate_seed_state(self, N=None):
		if N is None:
			N = self.start_seed_len
		stream = self.rng.randint(low=1 << 16, high=1 << 31, size=N, dtype=np.uint32)
		return rans.unflatten(stream)


	def state_to_bytes(self, state):
		nums = rans.flatten(state)
		return bytes(bytearray(nums))


	def bytes_to_state(self, data):
		nums = np.frombuffer(data, dtype=np.uint32)

		# state = tuple(np.int32(int.from_bytes(x, byteorder='little', signed=True)) for x in data)
		return rans.unflatten(nums)


	# def set_state(self, state=None):
	# 	if state is None:
	# 		state = self.generate_seed_state()
	# 	self.state = state
	# 	return state


	def count_bits(self, state):
		# if state is None:
		# 	state = self.state
		# return 8 * (len(rans.flatten(state).tobytes()) - 4*self.start_seed_len)
		return 32 * (len(rans.flatten(state)) - self.start_seed_len)


	def compress_append(self, images, state=None):
		if state is None:
			state = self.generate_seed_state()
			# if self.state is None:
			# 	state = self.generate_seed_state()
		# counts = []
		for image in images:
			state = self.vae_append(state, image.cpu().float().unsqueeze(0)) #.mul(255).round()
			# counts.append(self.count_bits(state))
		# self.state = state
		return state#, counts


	def partial_decompress(self, state, N=None):
		imgs = []
		while (N is not None and N > 0) or (N is None and len(rans.flatten(state)) > self.start_seed_len):
			state, img = self.vae_pop(state)
			imgs.append(img)
			if N is not None:
				N -= 1
		# self.state = state
		imgs = torch.stack(imgs[::-1]).round().byte().view(-1, *self.enc_in_shape).cpu()#.div(255)
		return state, imgs


	def compress(self, images, state=None):
		if not self._as_bytes:
			images = images.mul(255).byte()
		state = self.compress_append(images, state)
		return self.state_to_bytes(state)


	def decompress(self, data):
		state = self.bytes_to_state(data)
		images = self.partial_decompress(state)[-1]
		if not self._as_bytes:
			images = images.float().div(255)
		return images




