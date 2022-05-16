import torch
from torch import nn
from torch.nn import functional as F
from piq.feature_extractors import InceptionV3 as _InceptionV3


from ...framework import Extractor, ModuleParametrized, hparam, inherit_hparams, spaces


class InceptionV3(Extractor, ModuleParametrized, _InceptionV3):
	requires_grad = False
	resize_input = True
	normalize_input = True
	use_fid_inception = True
	
	dim = hparam(2048)
	
	def __init__(self, dim=None, resize_input=None, normalize_input=None, use_fid_inception=None, requires_grad=False,
	             device='cuda' if torch.cuda.is_available() else None,
	             **kwargs):
		if dim is None:
			dim = self.dim
		if resize_input is None:
			resize_input = self.resize_input
		if normalize_input is None:
			normalize_input = self.normalize_input
		if use_fid_inception is None:
			use_fid_inception = self.use_fid_inception
		if requires_grad is None:
			requires_grad = self.requires_grad
		super().__init__(output_blocks=[self.BLOCK_INDEX_BY_DIM[dim]],
		                 resize_input=resize_input, normalize_input=normalize_input, requires_grad=requires_grad,
		                 use_fid_inception=use_fid_inception, **kwargs)
		if not resize_input:
			self.din = spaces.Pixels(3, 299, 299, as_bytes=False)
		self.dout = spaces.HalfBound(min=0, shape=(dim,))
		
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.eval()
		if device is not None:
			self.to(device)
		self.device = device
	
	
	def forward(self, x):
		if x.shape[1] == 1: # automatically convert greyscale into RGB
			x = torch.cat([x]*3, 1)
		if self.device is not None:
			x = x.to(self.device)
		z = self.pool(super().forward(x)[0]).squeeze(-1).squeeze(-1)
		return z








