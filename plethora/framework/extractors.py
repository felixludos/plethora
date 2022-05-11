from omnibelt import unspecified_argument
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import timm

# from ..framework import util
from ..framework import Extractor, Rooted, Device
from . import spaces


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


def get_inceptionV3(): # TODO: enable different block indices
	return Timm_Extractor('inceptionv3')


class Timm_Extractor(Rooted, Device, Extractor, nn.Module):
	def __init__(self, model_name=None, pool_features=True, din=None,
	             auto_resize=unspecified_argument, resize_mode='bilinear',
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

		if auto_resize is unspecified_argument and din is None:
			auto_resize = self._infer_auto_resize_inputs(model_name)
		if auto_resize is not None and isinstance(auto_resize, int):
			auto_resize = auto_resize, auto_resize
		self.auto_resize_input = auto_resize
		self.auto_resize_mode = resize_mode

		self._fix_channels = self.auto_resize_input is None
		self.din, self.dout = self._infer_dim(din)
		self.to(self.device)


	def _infer_auto_resize_inputs(self, model_name):
		return {'inception_v3': 299, }.get(model_name, 128)


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
		Cout = self.model.num_features
		if din is not None:
			Cin, Hin, Win = din.shape if isinstance(din, spaces.Dim) else din
			if Cin != 3:
				self._fix_channels = True
			factor = self.model.feature_info[-1]['reduction']
			if Hin is not None:
				Hout = int(np.ceil(Hin/factor))
			if Win is not None:
				Wout = int(np.ceil(Win/factor))

			din = din if isinstance(din, spaces.Image) else spaces.Image(Cin, Hin, Win)
			dout = spaces.Image(Cout, Hout, Wout)
		if self.pool:
			dout = spaces.Unbound(shape=(Cout,))
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
		if self.auto_resize_input is not None and x.shape[-2:] != self.auto_resize_input:
			x = F.interpolate(x, self.auto_resize_input, mode=self.auto_resize_mode)
		f = self.model.forward_features(x)
		if self.pool is not None:
			f = self.pool(f)
			f = f.view(f.size(0), -1)
		return f.to(device)




