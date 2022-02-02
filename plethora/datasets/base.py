import math
import torch

from omnibelt import unspecified_argument, duplicate_instance

from ..framework import base



class MissingModeError(Exception):
	pass



class BufferNotFoundError(Exception):
	pass



class TensorBuffer(base.FixedBuffer):
	def __init__(self, data=None, **kwargs):
		super().__init__(**kwargs)
		self.data = None
		self.set_data(data)


	def set_data(self, data=None):
		self.device = None if data is None else data.device
		self.data = data


	def is_loaded(self):
		return self.data is not None


	def _count(self):
		return len(self.data)


	def _to(self, device):
		self.data = self.data.to(device)


	def _load(self):
		pass


	def _get(self, idx, device=None, **kwargs):
		sample = self.data[idx]
		if device is not None:
			sample = sample.to(device)
		return sample


	def _update(self, indices=None, **kwargs):
		if indices is not None:
			self.data = self.data[indices]




class ReferencedBuffer(base.Buffer):
	def __init__(self, ref=None, **kwargs):
		super().__init__(**kwargs)
		self.ref = ref


	def _load(self, *args, **kwargs):
		pass


	def _get(self, idx, *args, **kwargs):
		return self._collection.buffers.get(self.ref, {}).get(idx)


	def _update(self, idx, *args, **kwargs):
		pass



class WrappedBuffer(base.Buffer):
	def __init__(self, source=None, space=None, **kwargs):
		super().__init__(space=None, **kwargs)
		self.source = source


	def set_source(self, source):
		self.source = source


	def set_space(self, space):
		self.space = space


	def get_space(self):
		if self.space is None:
			return self.source.get_space()
		return self.space


	def _load(self, *args, **kwargs):
		pass


	def _update(self, idx, *args, **kwargs):
		pass


	def _get(self, idx, device=None, **kwargs):
		return self.source.get(idx, device=device, **kwargs)



class DataCollection(base.Buffer):
	sample_format = None
	
	def __init__(self, *, sample_format=unspecified_argument, batch_device=unspecified_argument, mode=None,
	             buffers=None, modes=None, space=None, **kwargs):
		super().__init__(space=None, **kwargs)
		if sample_format is unspecified_argument:
			sample_format = self.sample_format
		self.set_sample_format(sample_format)

		if batch_device is unspecified_argument:
			batch_device = None
		self.batch_device = batch_device

		if buffers is None:
			buffers = {}
		self.buffers = buffers

		self._mode = mode
		if modes is None:
			modes = {}
			if mode is not None:
				modes[mode] = self
		self._modes = modes


	@staticmethod
	def _default_buffer_factory(name, **kwargs):
		raise NotImplementedError


	def register_buffer(self, name, buffer=None, space=unspecified_argument, **kwargs):
		if not isinstance(buffer, base.Buffer):
			buffer = self._default_buffer_factory(name, data=buffer, **kwargs)
		if space is not unspecified_argument:
			buffer.set_space(space)
		self.buffers[name] = buffer
		return self.buffers[name]

	def _remove_buffer(self, name):
		if self.has_buffer(name):
			del self.buffers[name]

	def rename_buffer(self, current, new=None):
		buffer = self.get_buffer(current)
		if buffer is not None:
			self._remove_buffer(current)
		if new is not None:
			self.register_buffer(new, buffer)


	def get_space(self, name):
		if name not in self.buffers:
			raise BufferNotFoundError(name)
		return self.buffers[name].get_space()


	def _load(self, *args, **kwargs):
		for name, buffer in self.buffers.items():
			buffer.load()


	def _check_buffer_names(self, *names):
		missing = []
		for name in names:
			if name not in self.buffers:
				missing.append(name)
		if len(missing):
			raise BufferNotFoundError(', '.join(missing))


	def get_buffer(self, name):
		return self.buffers.get(name)


	def has_buffer(self, name):
		return name in self.buffers


	def _get(self, idx=None, sample_format=None, device=None, strict=True):
		if sample_format is None:
			sample_format = self.sample_format
		if device is None:
			device = self.batch_device

		if isinstance(sample_format, str):
			if strict:
				self._check_buffer_names(sample_format)
			batch = self.buffers[sample_format].get(idx, device=device)
		elif isinstance(sample_format, (list, tuple)):
			if strict:
				self._check_buffer_names(*sample_format)
			batch = [self.buffers[name].get(idx, device=device) for name in sample_format]
		elif isinstance(sample_format, set):
			if strict:
				self._check_buffer_names(*sample_format)
			batch = {name:self.buffers[name].get(idx, device=device)
			         for name in sample_format if name in self.buffers}
		elif isinstance(sample_format, dict):
			if strict:
				self._check_buffer_names(*sample_format)
			batch = {}
			for name, transforms in sample_format.items():
				if name in self.buffers:
					samples = self.buffers[name].get(idx)
					if transforms is not None and not isinstance(transforms, (list, tuple)):
						transforms = [transforms]
					for transform in transforms:
						samples = transform.transform(samples)
					batch[name] = samples
		else:
			raise NotImplementedError(f'bad sample format: {sample_format}')

		return batch


	def set_sample_format(self, sample_format):
		self.sample_format = sample_format


	def get_sample_format(self):
		return self.sample_format


	def batch(self, N=None):
		raise NotImplementedError


	def register_modes(self, **modes):
		self._modes.update(modes)


	@property
	def mode(self):
		return self._mode


	def get_mode(self, mode='train'):
		if self.mode == mode:
			return self
		if mode in self._modes:
			return self._modes[mode]
		raise MissingModeError


	
class Dataset(DataCollection, base.FixedBuffer):
	def __init__(self, batch_size=64, shuffle_batches=True, **kwargs):
		super().__init__(**kwargs)

		self._batch_size = batch_size
		self._shuffle_batches = shuffle_batches

		self._waiting_subset = None
		self._subset_indices = None


	@staticmethod
	def _default_buffer_factory(name, **kwargs):
		return TensorBuffer(**kwargs)


	@staticmethod
	def _is_big_number(N):
		return N > 10000000


	@classmethod
	def shuffle_indices(cls, N, seed=None, generator=None):
		if seed is not None and generator is None:
			generator = torch.Generator()
			generator.manual_seed(seed)
		# TODO: include a warning if cls._is_big_number(N)
		return torch.randint(N, size=(N,), generator=generator) \
			if cls._is_big_number(N) else torch.randperm(N, generator=generator)


	@staticmethod
	def _split_indices(indices, cut):
		assert cut != 0
		last = cut < 0
		cut = abs(cut)
		total = len(indices)
		if isinstance(cut, float):
			assert 0 < cut < 1
			cut = int(cut * total)
		part1, part2 = indices[:cut], indices[cut:]
		if last:
			part1, part2 = part2, part1
		return part1, part2


	@staticmethod
	def _wrap_buffer(source, indices=None, **kwargs):
		return WrappedBuffer(source, indices=indices, **kwargs)


	def subset(self, cut=None, indices=None, shuffle=False):
		if indices is None:
			indices = self._split_indices(indices=self.shuffle_indices(len(self), generator=self.gen)
										  if shuffle else torch.arange(len(self)), cut=cut)
		new = self.copy()
		for name, buffer in self.buffers.items():
			new.register_buffer(name, self._wrap_buffer(buffer, indices))
		if self.mode is not None:
			self.register_modes(**{self.mode: new})
		return new


	def __iter__(self):
		raise NotImplementedError


	def __next__(self):
		raise NotImplementedError


	def _count(self):
		len(next(iter(self.buffers.values())))
	
	
	def split(self, splits, shuffle=False, register_modes=True):
		auto_name = isinstance(splits, (list, tuple, set))
		register_modes = register_modes and not auto_name
		if auto_name:
			named_cuts = [(f'part{i}', r) for i, r in enumerate(splits)]
		else:
			assert isinstance(splits, dict), f'unknown splits: {splits}'
			assert not any(x for x in splits if x is None), 'names of splits cannot be None'
			named_cuts = list(splits.items())
		names, cuts = zip(*sorted(named_cuts, key=lambda nr: (isinstance(nr[1], int), isinstance(nr[1], float),
		                                                          nr[1] is None, nr[0]), reverse=True))

		remaining = len(self)
		nums = []
		itr = iter(cuts)
		for cut in itr:
			if isinstance(cut, int):
				nums.append(cut)
				remaining -= cut
			else:
				if isinstance(cut, float):
					ratios = []
					while isinstance(cut, float):
						ratios.append(cut)
						cut = next(itr, 'done')
					if len(cuts):
						rationums = [remaining*abs(ratio) for ratio in ratios]
						nums.extend([math.copysign(1, r)*n for r, n in zip(ratios, rationums)])
						remaining -= sum(rationums)
				if cut is None:
					pieces = len([cut, *itr])
					assert remaining > pieces, f'cant evenly distribute {remaining} samples into {pieces} cuts'
					evennums = [remaining//pieces for _ in range(pieces)]
					nums.extend(evennums)
					remaining -= sum(evennums)

		if remaining > 0:
			nums[-1] += remaining

		indices = self.shuffle_indices(len(self), generator=self.gen) if shuffle else torch.arange(len(self))

		parts = {}
		for name, num in zip(names, nums):
			part, indices = self._split_indices(indices, num)
			parts[name] = part

		if register_modes:
			self.register_modes(**parts)

		if auto_name:
			return [parts[name] for name, _ in named_cuts]
		return parts




class DataGenerator(DataCollection): # TODO: batches are specified by number of samples, not a list of indicies

	def __iter__(self):
		raise NotImplementedError


	def __next__(self):
		raise NotImplementedError

	pass


class _DatasetIterator: # TODO: pbar integration
	def __init__(self, dataset, generator=None, seed=None, **kwargs):
		super().__init__(**kwargs)



		pass


class _DatagenIterator:
	pass



class ObservationDataset(Dataset):
	def get_observation_space(self):
		return self.get_space('observation')


	def get_observation(self, idx=None, **kwargs):
		return self.get(idx=idx, sample_format='observation', **kwargs)


	def __len__(self):
		return len(self.get_buffer('observation'))



class SupervisedDataset(ObservationDataset):
	def get_target_space(self):
		return self.get_space('target')

	def get_observation(self, idx=None, **kwargs):
		return self.get(idx=idx, sample_format='target', **kwargs)



class LabeledDataset(SupervisedDataset):
	def get_label_space(self):
		return self.get_space('label')



class DisentanglementDataset(LabeledDataset):
	def get_mechanism_space(self):
		raise NotImplementedError


	def transform_to_mechanisms(self, data):
		return self.get_mechanism_space().transform(data, self.get_label_space())


	def transform_to_labels(self, data):
		return self.get_label_space().transform(data, self.get_mechanism_space())


	def get_observations_from_labels(self, labels):
		raise NotImplementedError


	def difference(self, a, b, standardize=None):
		if standardize is None:
			standardize = self._standardize_scale
		if not self.uses_mechanisms():
			a, b = self.transform_to_mechanisms(a), self.transform_to_mechanisms(b)
		return self.get_mechanism_space().difference(a,b, standardize=standardize)


	def distance(self, a, b, standardize=None):
		if standardize is None:
			standardize = self._standardize_scale
		if not self.uses_mechanisms():
			a, b = self.transform_to_mechanisms(a), self.transform_to_mechanisms(b)
		return self.get_mechanism_space().distance(a,b, standardize=standardize)





