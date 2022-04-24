import sys, os, shutil
from omnibelt import unspecified_argument, agnosticmethod
import numpy as np
import torch
import subprocess
import wget
import h5py as hf

from ..framework.util import spaces
from .base import ImageDataset, SyntheticDataset
from .buffers import TransformedBuffer
from .mnist import ImageBuffer


class DownloadableHDF(ImageDataset):
	_image_key_name = 'images'


	def _prepare(self, *args, **kwargs):
		dest = self.get_archive_path()
		if not dest.is_file():
			if self._auto_download:
				self.download()
			else:
				raise self.DatasetNotDownloaded
		super()._prepare(*args, **kwargs)


	@staticmethod
	def _download_source_hdf(dest):
		raise NotImplementedError


	@agnosticmethod
	def get_archive_path(self):
		return self.get_root() / f'{self.name}.h5'


	@agnosticmethod
	def download(self, testset_ratio=0.2, test_seed=0, force_download=False, **kwargs):
		dest = self.get_archive_path()
		if not dest.is_file() or force_download:
			self._download_source_hdf(dest)

		rng = np.random.RandomState(test_seed)
		assert testset_ratio is not None and testset_ratio > 0, 'bad testset ratio'

		with hf.File(dest, 'r+') as f:
			N = f[self._image_key_name].shape[0]

			test_N = int(N * testset_ratio)

			order = rng.permutation(N)

			train_idx = order[:-test_N]
			train_idx.sort()
			test_idx = order[-test_N:]
			test_idx.sort()

			f.create_dataset('train_idx', data=train_idx)
			f.create_dataset('test_idx', data=test_idx)



class dSprites(DownloadableHDF, SyntheticDataset):
	name = 'dsprites'
	def __init__(self, default_len=None, as_bytes=False, **kwargs):
		# if default_len is None:
		# 	default_len = 737280
		super().__init__(default_len=default_len, **kwargs)

		self.register_buffer('observation', ImageBuffer(),
		                     space=spaces.Pixels(1, 64, 64, as_bytes=as_bytes))

		_shape_names = ['square', 'ellipse', 'heart']
		_dim_names = ['shape', 'scale', 'orientation', 'posX', 'posY']
		self.register_buffer('label',
		                     space=spaces.JointSpace(spaces.Categorical(_shape_names),
		                                             spaces.Categorical(6),
		                                             spaces.Categorical(40),
		                                             spaces.Categorical(32),
		                                             spaces.Categorical(32),
		                                             names=_dim_names))
		self.register_buffer('mechanism', TransformedBuffer(source=self.get_buffer('label')),
		                     space=spaces.JointSpace(spaces.Categorical(_shape_names),
		                                             spaces.Bound(0.5, 1.),
		                                             spaces.Periodic(period=2 * np.pi),
		                                             spaces.Bound(0., 1.),
		                                             spaces.Bound(0., 1.),
		                                             names=_dim_names))


	_source_url = 'https://github.com/deepmind/dsprites-dataset/raw/master/' \
	              'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
	_image_key_name = 'imgs'


	@classmethod
	def _download_source_hdf(cls, dest):
		wget.download(cls._source_url, str(dest))


	@classmethod
	def _decode_meta_info(cls, obj):
		'''
		recursively convert bytes to str
		:param obj: root obj
		:return:
		'''
		if isinstance(obj, dict):
			return {cls._decode_meta_info(k): cls._decode_meta_info(v) for k, v in obj.items()}
		if isinstance(obj, list):
			return [cls._decode_meta_info(x) for x in obj]
		if isinstance(obj, tuple):
			return tuple(cls._decode_meta_info(x) for x in obj)
		if isinstance(obj, bytes):
			return obj.decode()
		return obj


	def _prepare(self, *args, **kwargs):
		super()._prepare(*args, **kwargs)

		dest = self.get_archive_path()

		# data = np.load(path, allow_pickle=True, encoding='bytes')
		# self.meta = self._decode_meta_info(data['metadata'][()])

		with hf.File(str(dest), 'r') as f:
			self.get_buffer('observation').data = torch.from_numpy(f[self._image_key_name][()]).unsqueeze(1)
			# self.get_buffer('label').data = torch.from_numpy(data['latents_values'][:, 1:]).float()
			self.get_buffer('label').data = torch.from_numpy(f['latents_classes'][:, 1:]).float()



class Shapes3D(DownloadableHDF, SyntheticDataset):
	name = '3dshapes'
	_default_lens = {'train': 384000, 'test': 96000, 'full': 480000}

	def __init__(self, default_len=None, as_bytes=False, mode=None, **kwargs):
		if default_len is None:
			if mode is None:
				mode = 'train'
			default_len = self._default_lens[mode]
		super().__init__(default_len=default_len, **kwargs)

		_all_label_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
		_hue_names = ['red', 'orange', 'yellow', 'green', 'seagreen', 'cyan', 'blue', 'dark-blue', 'purple', 'pink']
		_shape_names = ['cube', 'cylinder', 'ball', 'capsule']

		self.register_buffer('observation', ImageBuffer(),
		                     space=spaces.Pixels(3, 64, 64, as_bytes=as_bytes))
		self.register_buffer('mechanism',
		                     space=spaces.JointSpace(spaces.Periodic(),
		                                             spaces.Periodic(),
		                                             spaces.Periodic(),
		                                             spaces.Bound(0.75, 1.25),
		                                             spaces.Categorical(_shape_names),
		                                             spaces.Bound(-30., 30.),
		                                             names=_all_label_names))
		self.register_buffer('label', TransformedBuffer(source=self.get_buffer('mechanism')),
		                     space=spaces.JointSpace(spaces.Categorical(_hue_names),
		                                             spaces.Categorical(_hue_names),
		                                             spaces.Categorical(_hue_names),
		                                             spaces.Categorical(8),
		                                             spaces.Categorical(_shape_names),
		                                             spaces.Categorical(15),
		                                             names=_all_label_names))


	_source_url = 'gs://3d-shapes/3dshapes.h5'
	_image_key_name = 'images'


	@agnosticmethod
	def download(self, uncompress=False, **kwargs):
		super().download(**kwargs)

		if uncompress: # TODO: add logging
			arch = self.get_archive_path()
			new = arch.parents[0] / 'uncompressed.h5'

			with hf.File(str(new), 'w') as f:
				with hf.File(str(arch), 'r') as old:
					for key in old.keys():
						f.create_dataset(key, data=old[key][()])
					for key in old.attrs:
						f.attrs[key] = old.attrs[key]

			os.remove(str(arch))
			os.rename(str(new), str(arch))


	@classmethod
	def _download_source_hdf(cls, dest):
		# print(f'Downloading 3dshapes dataset to {str(dest)} ...', end='')
		subprocess.run(['gsutil', 'cp', cls._source_url, str(dest)])
		# print(' done!')


	def _prepare(self, *args, **kwargs):
		super()._prepare(*args, **kwargs)

		mode = self.mode

		dest = self.get_archive_path()
		with hf.File(str(dest), 'r') as f:
			indices = None if mode == 'full' else \
				(f['test_idx'][()] if mode == 'test' else f['train_idx'][()])

			images = f[self._image_key_name][()]
			mechanism = f['labels'][()]
			if indices is not None:
				images = images[indices]
				mechanism = mechanism[indices]

			# images = f[self._image_key_name][indices] # TODO: why is h5py so slow with indexed reads
			images = images.transpose(0, 3, 1, 2)

			self.get_buffer('observation').data = torch.from_numpy(images)
			self.get_buffer('mechanism').data = torch.from_numpy(mechanism).float()



class MPI3D(ImageDataset, SyntheticDataset):
	name = 'mpi3d'

	def __init__(self, cat='toy', default_len=None, as_bytes=False, **kwargs):
		# if default_len is None:
		# 	default_len = 480000
		super().__init__(default_len=default_len, **kwargs)
		if cat == 'sim':
			cat = 'realistic'
		assert cat in {'toy', 'realistic', 'real', 'complex'}, f'invalid category: {cat}'
		self.category = cat

		_all_label_names = ['object_color', 'object_shape', 'object_size', 'camera_height', 'background_color',
		                    'horizonal_axis', 'vertical_axis']
		_colors = ['white', 'green', 'red', 'blue', 'brown', 'olive']
		_shapes = ['cone', 'cube', 'cylinder', 'hexagonal', 'pyramid', 'sphere']
		_bg_color = ['purple', 'sea_green', 'salmon']
		if cat == 'complex':
			_shapes = ['mug', 'ball', 'banana', 'cup']
			_colors = ['yellow', 'green', 'olive', 'red']

		self.register_buffer('observation', ImageBuffer(),
		                     space=spaces.Pixels(3, 64, 64, as_bytes=as_bytes))

		self.register_buffer('label',
		                     space=spaces.JointSpace(spaces.Categorical(_colors),
		                                             spaces.Categorical(_shapes),
		                                             spaces.Categorical(['small', 'large']),
		                                             spaces.Categorical(['top', 'center', 'bottom']),
		                                             spaces.Categorical(_bg_color),
		                                             spaces.Categorical(40),
		                                             spaces.Categorical(40),
		                                             names=_all_label_names))
		self.register_buffer('mechanism', TransformedBuffer(source=self.get_buffer('label')),
		                     space=spaces.JointSpace(spaces.Categorical(_colors),
		                                             spaces.Categorical(_shapes),
		                                             spaces.Bound(0., 1.),
		                                             spaces.Bound(0., 1.),
		                                             spaces.Categorical(_bg_color),
		                                             spaces.Bound(0., 1.),
		                                             spaces.Bound(0., 1.),
		                                             names=_all_label_names))


	_source_url = {
		'toy': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz',
		'realistic': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz',
		'real': 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz',
	}
	@classmethod
	def download(cls, category=None, **kwargs):

		raise NotImplementedError

		if dataroot is None:
			dataroot = util.get_data_dir(A)
			dataroot = dataroot / 'mpi3d'
		dataroot.mkdir(exist_ok=True)

		if cat is None:
			cat = A.pull('category', 'toy')

		assert cat in {'toy', 'sim', 'realistic', 'real'}, f'invalid category: {cat}'
		if cat == 'sim':
			cat = 'realistic'

		ratio = A.pull('separate-testset', 0.2)

		path = dataroot / f'mpi3d_{cat}.h5'

		if not path.exists():
			rawpath = dataroot / f'mpi3d_{cat}.npz'
			if not rawpath.exists():
				print(f'Downloading mpi3d-{cat}')
				wget.download(cls._source_url[cat], str(rawpath))

			print('Loading full dataset into memory to split train/test')
			full = np.load(rawpath)['images']

			N = len(full)

			test_N = int(N * ratio)

			rng = np.random.RandomState(0)
			order = rng.permutation(N)

			train_idx = order[:-test_N]
			train_idx.sort()
			test_idx = order[-test_N:]
			test_idx.sort()

			with hf.File(path, 'w') as f:
				f.create_dataset('train_idx', data=train_idx)
				f.create_dataset('train_images', data=full[train_idx])

				f.create_dataset('test_idx', data=test_idx)
				f.create_dataset('test_images', data=full[test_idx])

			os.remove(str(rawpath))

















