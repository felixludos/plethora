
import numpy as np
import torch
from torch import nn
import timm

# from ..framework import util
from ..framework import Extractor, Rooted, Device
from ..framework.util import spaces


# class Extractor(nn.Module):
# 	def get_encoder_fingerprint(self):
# 		raise NotImplementedError
#
#
# 	def encode(self, x):
# 		return self(x)
#
#
# 	def extract(self, x):
# 		return self.encode(x)



class Timm_Extractor(Extractor, Rooted, Device, nn.Module):
	def __init__(self, model_name=None, pool_features=True, din=None,
	             pretrained=True, checkpoint_path='', global_pool='avg', drop_rate=0.,
	             create_kwargs=None, **kwargs):

		super().__init__(din=din, **kwargs)
		self.model_name = model_name

		if create_kwargs is None:
			create_kwargs = {}
		create_kwargs['model_name'] = model_name
		create_kwargs['pretrained'] = create_kwargs.get('pretrained', pretrained)
		create_kwargs['global_pool'] = create_kwargs.get('global_pool', global_pool)
		create_kwargs['drop_rate'] = create_kwargs.get('drop_rate', drop_rate)
		self._create_kwargs = create_kwargs
		create_kwargs['checkpoint_path'] = checkpoint_path

		model, pool = None, None
		if model_name is not None:
			model = timm.create_model(**create_kwargs)
			pool = self._find_global_pool(model) if pool_features else None
		self.model = model
		self.pool = pool

		self._fix_channels = False
		self.din, self.dout = self._infer_dim(din)
		self.to(self.device)


	def _to(self, device, **kwargs):
		return super(Device, self).to(device)


	def extract(self, observation):
		return self(observation)


	def get_extractor_key(self):
		return self._create_kwargs.copy()


	@classmethod
	def _find_global_pool(cls, model):
		for n, c in model.named_children():
			if n == 'global_pool':
				return c
			else:
				out = cls._find_global_pool(c)
				if out is not None:
					return out


	def _infer_dim(self, din=None):
		dout = None
		if din is not None:
			Cin, Hin, Win = din.shape if isinstance(din, spaces.Dim) else din
			if Cin != 3:
				self._fix_channels = True
			factor = self.model.feature_info[-1]['reduction']
			if Hin is not None:
				Hout = int(np.ceil(Hin/factor))
			if Win is not None:
				Wout = int(np.ceil(Win/factor))
			Cout = self.model.num_features
			din = din if isinstance(din, spaces.Image) else spaces.Image(Cin, Hin, Win)
			dout = spaces.Unbound(shape=(Cout,)) if self.pool is not None \
				else spaces.Image(Cout, Hout, Wout)
		return din, dout


	def get_classifier(self):
		return self.model.get_classifier()


	def forward(self, x):
		device = x.device
		x = x.to(self.device)
		if self._fix_channels and x.shape[1] != 3:
			if x.shape[1] == 1:
				x = torch.cat([x]*3, 1)
			else:
				assert x.shape[1] > 3, f'bad shape: {x.shape}'
				x = x[:, :3]
		f = self.model.forward_features(x)
		if self.pool is not None:
			f = self.pool(f)
			f = f.view(f.size(0), -1)
		return f.to(device)




