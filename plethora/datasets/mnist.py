
import numpy as np
import torch
from torch.nn import functional as F
import torchvision

from .base import Dataset, SupervisedDataset, ImageDataset, TensorBuffer
from ..framework.base import Buffer, FixedBuffer
from ..framework.util import spaces
from ..framework import Fileable



class ImageBuffer(TensorBuffer):
	def _get(self, *args, **kwargs):
		out = super()._get(*args, **kwargs)
		if not self.space.as_bytes:
			out = out.float().div(255)
		return out



class Torchvision_Toy_Dataset(SupervisedDataset, ImageDataset):
	def __init__(self, resize=True, download=None, target_attr='targets', _source_type=None, _source_kwargs=None,
	             _observation_space=None, _target_space=None, **kwargs):
		super().__init__(**kwargs)

		self._source_type = _source_type
		if _source_kwargs is None:
			_source_kwargs = {}
		self._source_kwargs = _source_kwargs

		if resize and _observation_space is not None:
			_observation_space.width = 32
			_observation_space.height = 32
		self.register_buffer('observation', ImageBuffer(), space=_observation_space)
		self.register_buffer('target', space=_target_space)

		self.resize = resize
		self._download = download
		self._target_attr = target_attr


	def _get_source_kwargs(self):
		kwargs = self._source_kwargs.copy()
		if 'root' not in kwargs:
			kwargs['root'] = self.get_root()
		if self.mode is not None:
			kwargs['train'] = self.mode != 'test'
		if 'download' not in kwargs and self._download is not None:
			kwargs['download'] = self._download
		return kwargs


	def _load(self, *args, **kwargs):
		src = self._source_type(**self._get_source_kwargs())

		images = src.data
		if isinstance(images, np.ndarray):
			images = torch.as_tensor(images)
		if images.ndimension() == 3:
			images = images.unsqueeze(1)
		if images.size(1) not in {1,3}:
			images = images.permute(0,3,1,2)
		if self.resize:
			images = F.interpolate(images.float(), (32, 32), mode='bilinear').round().byte()
		self.buffers['observation'].set_data(images)

		targets = getattr(src, self._target_attr)
		if not isinstance(targets, torch.Tensor):
			targets = torch.as_tensor(targets)
		self.buffers['target'].set_data(targets)

		super()._load(*args, **kwargs)



class MNIST(Torchvision_Toy_Dataset):
	name = 'mnist'
	def __init__(self, as_bytes=False, _source_type=None, _observation_space=None, _target_space=None, **kwargs):
		if _source_type is None:
			_source_type = torchvision.datasets.MNIST
		if _observation_space is None:
			_observation_space = spaces.PixelSpace(1, 28, 28, as_bytes=as_bytes)
		if _target_space is None:
			_target_space = spaces.CategoricalDim(10)
		super().__init__(_observation_space=_observation_space, _target_space=_target_space, **kwargs)



class KMNIST(Torchvision_Toy_Dataset):
	name = 'kmnist'
	def __init__(self, as_bytes=False, _source_type=None, _observation_space=None, _target_space=None, **kwargs):
		if _source_type is None:
			_source_type = torchvision.datasets.KMNIST
		if _observation_space is None:
			_observation_space = spaces.PixelSpace(1, 28, 28, as_bytes=as_bytes)
		if _target_space is None:
			_target_space = spaces.CategoricalDim(['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を'])
		super().__init__(_observation_space=_observation_space, _target_space=_target_space, **kwargs)



class FashionMNIST(Torchvision_Toy_Dataset):
	name = 'fmnist'
	def __init__(self, as_bytes=False, _source_type=None, _observation_space=None, _target_space=None, **kwargs):
		if _source_type is None:
			_source_type = torchvision.datasets.FashionMNIST
		if _observation_space is None:
			_observation_space = spaces.PixelSpace(1, 28, 28, as_bytes=as_bytes)
		if _target_space is None:
			_target_space = spaces.CategoricalDim(['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
			                                       'sneaker', 'bag', 'boot'])
		super().__init__(_observation_space=_observation_space, _target_space=_target_space, **kwargs)



class EMNIST(Torchvision_Toy_Dataset):
	name = 'emnist'
	def __init__(self, split='letters', as_bytes=False, _source_type=None,
	             _observation_space=None, _target_space=None, **kwargs):
		if _source_type is None:
			_source_type = torchvision.datasets.EMNIST
		if _observation_space is None:
			_observation_space = spaces.PixelSpace(1, 28, 28, as_bytes=as_bytes)
		if _target_space is None:
			assert split in _source_type.classes_split_dict, f'{split} vs {list(_source_type.classes_split_dict)}'
			_target_space = spaces.CategoricalDim(_source_type.classes_split_dict[split])
		super().__init__(_observation_space=_observation_space, _target_space=_target_space, **kwargs)
		self._split = split



class SVHN(Torchvision_Toy_Dataset):
	name = 'svhn'
	def __init__(self, as_bytes=False, target_attr=None, _source_type=None,
	             _observation_space=None, _target_space=None, **kwargs):
		if _source_type is None:
			_source_type = torchvision.datasets.SVHN
		if _observation_space is None:
			_observation_space = spaces.PixelSpace(3, 32, 32, as_bytes=as_bytes)
		if _target_space is None:
			_target_space = spaces.CategoricalDim(10)
		super().__init__(target_attr='labels',
		                 _observation_space=_observation_space, _target_space=_target_space, **kwargs)


	def _get_source_kwargs(self):
		kwargs = super()._get_source_kwargs()
		kwargs['split'] = 'train' if kwargs['train'] else 'test'
		del kwargs['train']
		return kwargs



class CIFAR10(Torchvision_Toy_Dataset):
	name = 'cifar10'
	def __init__(self, as_bytes=False, _source_type=None, _observation_space=None, _target_space=None, **kwargs):
		if _source_type is None:
			_source_type = torchvision.datasets.CIFAR10
		if _observation_space is None:
			_observation_space = spaces.PixelSpace(3, 32, 32, as_bytes=as_bytes)
		if _target_space is None:
			_target_space = spaces.CategoricalDim(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
			                                       'horse', 'ship', 'truck'])
		super().__init__(_observation_space=_observation_space, _target_space=_target_space, **kwargs)



class CIFAR100(Torchvision_Toy_Dataset):
	name = 'cifar100'
	def __init__(self, as_bytes=False, _source_type=None, _observation_space=None, _target_space=None, **kwargs):
		if _source_type is None:
			_source_type = torchvision.datasets.CIFAR100
		if _observation_space is None:
			_observation_space = spaces.PixelSpace(3, 32, 32, as_bytes=as_bytes)
		if _target_space is None:
			_target_space = spaces.CategoricalDim(
				['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
	                    'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
	                    'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
	                    'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
	                    'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower',
	                    'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
	                    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
	                    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
	                    'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
	                    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
	                    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
	                 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
				 ])
		super().__init__(_observation_space=_observation_space, _target_space=_target_space, **kwargs)









