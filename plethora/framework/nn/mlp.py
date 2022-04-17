import numpy as np
from torch import nn
from omnibelt import unspecified_argument, agnosticmethod

from ..util import spaces
from ..base import Function
from ..hyperparameters import hparam, inherit_hparams
from ..models import Model


class Sequential(Model, Function, nn.Sequential):


	nonlin = hparam('elu', space=['elu', 'relu', ])



	class Builder(Model.Builder):

		def __init__(self, din, dout, hidden=(), nonlin='elu', output_nonlin=None,
		             norm=None, output_norm=None, bias=True, output_bias=None, **kwargs):
			layers = self._create_layers(din, dout, hidden=hidden, nonlin=nonlin, output_nonlin=output_nonlin,
			                             norm=norm, output_norm=output_norm, bias=bias, output_bias=output_bias)
			super().__init__(*layers, din=din, dout=dout, **kwargs)

		@agnosticmethod
		def _expand_dim(self, dim):
			if isinstance(dim, int):
				dim = [dim]
			if isinstance(dim, (list, tuple)):
				return dim
			if isinstance(dim, spaces.DimSpec):
				return dim.expanded_shape
			raise NotImplementedError(dim)


		def _build_layer(self, din, dout, nonlin='elu', norm=None, bias=True):
			in_shape = self._expand_dim(din)
			in_width = int(np.product(in_shape))

			layers = []

			if len(in_shape) > 1:
				layers.append(nn.Flatten())

			out_shape = self._expand_dim(dout)
			out_width = int(np.product(in_shape))

			layers.append(nn.Linear(in_width, out_width, bias=bias))
			if nonlin is not None:
				layers.append(self._create_nonlin())

			return dout, [layer]


		@agnosticmethod
		def _build_outlayer(self, din, dout, nonlin=None, norm=None, bias=True):
			shape = self._expand_dim(dout)
			return []


		@agnosticmethod
		def _create_layers(self, din, dout, hidden=(),
		             nonlin='elu', output_nonlin=None,
		             norm=None, output_norm=None,
		             bias=True, output_bias=None):

			if output_bias is None:
				output_bias = bias

			flatten = False
			reshape = None

			din = din
			dout = dout

			if isinstance(din, (tuple, list)):
				flatten = True
				din = int(np.product(din))
			if isinstance(dout, (tuple, list)):
				reshape = dout
				dout = int(np.product(dout))

			nonlins = [nonlin] * len(hidden) + [output_nonlin]
			norms = [norm] * len(hidden) + [output_norm]
			biases = [bias] * len(hidden) + [output_bias]
			hidden = din, *hidden, dout

			layers = []
			if flatten:
				layers.append(nn.Flatten())

			for in_dim, out_dim, nonlin, norm, bias in zip(hidden, hidden[1:], nonlins, norms, biases):
				layer = nn.Linear(in_dim, out_dim, bias=bias)
				if initializer is not None:
					layer = initializer(layer, nonlin)
				layers.append(layer)
				if norm is not None:
					layers.append(util.get_normalization1d(norm, out_dim))
				if nonlin is not None:
					layers.append(util.get_nonlinearity(nonlin))

			if reshape is not None:
				layers.append(Reshaper(reshape))

			pass







