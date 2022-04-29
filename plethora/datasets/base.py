import json
import os
from pathlib import Path
import math
from collections import OrderedDict
import torch
from omnibelt import unspecified_argument, duplicate_instance, md5, agnosticmethod
import h5py as hf

from ..framework.features import Prepared
from ..framework import base, Rooted, Named, util#, Device
from .buffers import AbstractFixedBuffer, Buffer, BufferView, HDFBuffer, \
	AbstractCountableData, AbstractCountableDataView



class Batchable(base.AbstractData):
	def get_iterator(self, infinite=False, **kwargs):
		first = True
		while first or infinite:
			for sel in self.generate_selections(**kwargs):
				yield self.create_batch(sel=sel)
			first = False


	def __iter__(self):
		return self.get_iterator()


	def get_batch(self, **kwargs):
		return next(self.get_iterator(**kwargs))


	def __next__(self):
		return self.get_batch()


	def generate_selections(self, **kwargs):
		raise NotImplementedError


	class NoBatch(base.AbstractData.NoView):
		pass


	Batch = None
	def create_batch(self, sel=None, **kwargs):
		if self.Batch is None:
			raise self.NoBatch
		return self.Batch(source=self, sel=sel, **kwargs)
	


class Epoched(AbstractCountableData, Batchable, base.Seeded): # TODO: check Seeded and Device integration
	'''Batchable with a fixed total number of samples (implements __len__)'''
	def __init__(self, batch_size=64, shuffle_batches=True, force_batch_size=True,
	             # batch_device=None,
	             infinite=False, **kwargs):
		super().__init__(**kwargs)

		self._batch_size = batch_size
		self._force_batch_size = force_batch_size
		self._shuffle_batches = shuffle_batches
		# self._batch_device = batch_device
		self._infinite = infinite


	# def create_batch(self, sel=None, device=unspecified_argument, **kwargs):
	# 	if device is unspecified_argument:
	# 		device = self._batch_device
	# 	return super().create_batch(sel=sel, device=device, **kwargs)


	def get_batch(self, shuffle=None, **kwargs):
		if shuffle is None:
			shuffle = True
		return next(self.get_iterator(shuffle=shuffle, **kwargs))


	@property
	def batch_size(self):
		return self._batch_size


	@staticmethod
	def _is_big_number(N):
		return N > 10000000
	
	
	@classmethod
	def shuffle_indices(cls, N, gen=None):
		# if seed is not None and gen is None:
		# 	gen = torch.Generator()
		# 	gen.manual_seed(seed)
		# TODO: include a warning if cls._is_big_number(N)
		return torch.randint(N, size=(N,), generator=gen) \
			if cls._is_big_number(N) else torch.randperm(N, generator=gen)


	def generate_selections(self, sel=None, num_samples=None, batch_size=None, shuffle=False,
	                        force_batch_size=None, gen=None, **kwargs):
		if batch_size is None:
			batch_size = self.batch_size
		if force_batch_size is None:
			force_batch_size = self._force_batch_size
		if gen is None:
			gen = self.gen
			
		if sel is None:
			sel = torch.arange(self.size())
		if shuffle:
			sel = sel[self.shuffle_indices(len(sel), gen=gen)]
		order = sel
		if num_samples is not None and len(order) > num_samples:
			order = order[:max(num_samples, batch_size) if force_batch_size else num_samples]
		inds = list(order.split(batch_size))
		if force_batch_size and len(inds) and len(inds[-1]) != batch_size:
			inds.pop()
		return inds
	

	def get_iterator(self, epochs=1, num_samples=None, num_batches=None, infinite=False, hard_limit=True,
	                 batch_size=None, shuffle=None, force_batch_size=None, gen=None, sel=None, pbar=None, **kwargs):
		if batch_size is None:
			batch_size = self.batch_size
		if force_batch_size is None:
			force_batch_size = self._force_batch_size
		if shuffle is None:
			shuffle = self._shuffle_batches
		if gen is None:
			gen = self.gen
			
		subsel = sel

		N = self.size() if subsel is None else len(subsel)
		samples_per_epoch = N - int(force_batch_size) * (N % batch_size)
		batches_per_epoch = int(math.ceil(samples_per_epoch / batch_size))
		if infinite is None:
			total_samples = None
		elif num_batches is not None:
			total_samples = (num_batches % batches_per_epoch) * batch_size \
			                + (num_batches // batches_per_epoch) * samples_per_epoch
		elif num_samples is not None:
			total_samples = samples_per_epoch * (num_samples // samples_per_epoch)
			remainder = num_samples % samples_per_epoch
			total_samples += batch_size * (remainder // batch_size)
			remainder = remainder % batch_size
			if not hard_limit or not force_batch_size:
				total_samples += remainder
		else:
			total_samples = samples_per_epoch * epochs
		if pbar is not None:
			pbar = pbar(total=total_samples)
			
		while total_samples is None or total_samples > 0:
			sels = self.generate_selections(sel=subsel, num_samples=total_samples, batch_size=batch_size,
			                                shuffle=shuffle, force_batch_size=force_batch_size, gen=gen, **kwargs)
			for sel in sels:
				N = len(sel)
				if total_samples is not None:
					total_samples -= N
					if hard_limit and total_samples < 0:
						break
				if pbar is not None:
					pbar.update(N)
				yield self.create_batch(sel=sel)
				if total_samples is not None and total_samples <= 0:
					break
		if pbar is not None:
			pbar.close()



class DataSource(Batchable, base.AbstractData, Named):
	class MissingBuffer(Exception):
		pass


	def available_buffers(self): # returns list of names
		raise NotImplementedError


	def iter_buffers(self): # iterates through buffers
		raise NotImplementedError


	def register_buffer(self, name, buffer, **kwargs):
		raise NotImplementedError


	def get_buffer(self, name):
		raise NotImplementedError


	def has_buffer(self, name):
		raise NotImplementedError


	def __getitem__(self, name):
		return self.get(name)


	def __contains__(self, item):
		return self.has_buffer(item)


	def space_of(self, name):
		return self.get_buffer(name).space


	def get(self, name, sel=None, **kwargs):
		return super().get(sel=sel, name=name, **kwargs)


	def _get(self, name, sel=None, **kwargs):
		return self.get_buffer(name).get(sel)
		buffer = self.get_buffer(name)
		data = buffer.get(sel)
		return data


	def _prepare(self, *args, **kwargs):
		for name, buffer in self.iter_buffers(True):
			buffer.prepare()


	def _title(self):
		return self.name



class SourceView(DataSource, base.AbstractView):
	# _is_ready = True

	View = None


	def _title(self):
		return '' if self.source is None else self.source._title()


	def __str__(self):
		src = '{' + self._title() + '}'
		return f'{self.__class__.__name__}{src}'#"{hex(id(self))[2:]}"'


	# def get(self, name, sel=unspecified_argument, **kwargs):
	# 	if sel is unspecified_argument:
	# 		sel = self.sel
	# 	return super().get(sel=sel, name=name, **kwargs)


	def _get(self, name, sel=None, **kwargs):
		if self.source is None:
			raise self.NoSource
		sel = self._merge_sel(sel)
		return self.source.get(name, sel=sel, **kwargs)
		# return self.get_buffer(name).get(sel)
		buffer = self.get_buffer(name)
		data = buffer.get(sel)
		return data


	def _update(self, sel=None, **kwargs):
		if self.source is None:
			raise self.NoSource
		sel = self._merge_sel(sel)
		return self.source.update(sel=sel, **kwargs)


	Batch = None
	def create_batch(self, sel=None, **kwargs):
		if self.Batch is None:
			if self.source is None:
				raise self.NoSource
			return self.source.create_batch(**kwargs)
		return super().create_batch(**kwargs)


	def available_buffers(self): # returns list of names
		if self.source is None:
			raise self.NoSource
		return self.source.available_buffers()


	def get_buffer(self, name):
		if self.source is None:
			raise self.NoSource
		return self.source.get_buffer(name)


	def has_buffer(self, name):
		if self.source is None:
			raise self.NoSource
		return self.source.has_buffer(name)



DataSource.View = SourceView



class MultiModed(DataSource):
	def __init__(self, *, mode=None, modes=None, **kwargs):
		super().__init__(**kwargs)
		if modes is None:
			modes = OrderedDict()
		self._modes = modes
		self._mode = mode


	def copy(self):
		new = super().copy()
		new._modes = new._modes.copy()
		return new


	def register_modes(self, **modes):
		for name, mode in modes.items():
			mode._mode = name
		self._modes.update(modes)


	@property
	def mode(self):
		return self._mode


	class MissingModeError(Exception):
		pass


	def get_mode(self, mode='train'):
		if self.mode == mode:
			return self
		if mode in self._modes:
			return self._modes[mode]
		raise self.MissingModeError



class BufferTable(DataSource):
	def __init__(self, buffers=None, **kwargs):
		super().__init__(**kwargs)
		if buffers is None:
			buffers = OrderedDict()
		self.buffers = buffers


	def available_buffers(self): # returns list of names
		return list(self.buffers.keys())


	def iter_buffers(self, items=True): # iterates through buffers
		for k, v in self.buffers.items():
			if not isinstance(v, str):
				yield (k,v) if items else v


	def get_buffer(self, name):
		if name not in self.buffers:
			raise self.MissingBuffer(name)
		buffer = self.buffers[name]
		if isinstance(buffer, str):
			return self.get_buffer(buffer)
		return buffer


	def has_buffer(self, name):
		return name in self.buffers


	def copy(self):
		new = super().copy()
		new.buffers = new.buffers.copy()
		return new


	class InvalidBuffer(Exception):
		def __init__(self, name, buffer):
			super().__init__(f'{name}: {buffer}')
			self.name, self.buffer = name, buffer


	Buffer = Buffer
	@classmethod
	def _create_buffer(cls, **kwargs):
		return cls.Buffer(**kwargs)


	def register_buffer(self, name, buffer=None, space=unspecified_argument, **kwargs):
		if isinstance(buffer, str):
			assert space is unspecified_argument, 'cant specify a space for an alias'
		elif not isinstance(buffer, base.AbstractBuffer):
		# elif buffer is None or isinstance(buffer, torch.Tensor):
			if type(buffer) == type and issubclass(buffer, base.AbstractBuffer):
				if space is not unspecified_argument:
					kwargs['space'] = space
				buffer = buffer(**kwargs)
			else:
				kwargs['data'] = buffer
				if space is not unspecified_argument:
					kwargs['space'] = space
				buffer = self._create_buffer(**kwargs)
		if space is not unspecified_argument:
			buffer.space = space
		if not isinstance(buffer, str) and not self._check_buffer(name, buffer):
			raise InvalidBuffer(name, buffer)
		self.buffers[name] = buffer
		return self.buffers[name]


	def _check_buffer(self, name, buffer): # during registration
		return True


	def _remove_buffer(self, name):
		if name in self.buffers:
			del self.buffers[name]


	def rename_buffer(self, current, new=None):
		buffer = self.get_buffer(current)
		if buffer is not None:
			self._remove_buffer(current)
		if new is not None:
			self.register_buffer(new, buffer)



class ReplacementView(BufferTable, SourceView):
	def available_buffers(self):
		buffers = super().available_buffers()
		for replacement in super(BufferTable, self).available_buffers():
			if replacement not in buffers:
				buffers.append(replacement)
		return buffers


	def get_buffer(self, name):
		if name in self.buffers:
			return super().get_buffer(name)
		return super(BufferTable, self).get_buffer(name)


	def _get(self, name, sel=None, **kwargs):
		if name in self.buffers:
			return super(SourceView, self)._get(name, sel=sel, **kwargs)
		return super()._get(name, sel=sel, **kwargs)
		if self.source is None:
			raise self.NoSource
		sel = self._merge_sel(sel)
		return self.source.get(name, sel=sel, **kwargs)
		# return self.get_buffer(name).get(sel)
		buffer = self.get_buffer(name)
		data = buffer.get(sel)
		return data


	def _update(self, sel=None, **kwargs):
		if name in self.buffers:
			return super(SourceView, self)._update(name, sel=sel, **kwargs)
		if self.source is None:
			raise self.NoSource
		sel = self._merge_sel(sel)
		return self.source.update(sel=sel, **kwargs)


	def has_buffer(self, name):
		return name in self.buffers or name in super(BufferTable, self).has_buffer(name)



class CountableView(AbstractCountableDataView, Epoched, SourceView):
	def get_iterator(self, *, sel=None, **kwargs):
		sel = self._merge_sel(sel)
		return super().get_iterator(sel=sel, **kwargs)


	def generate_selections(self, *, sel=None, **kwargs):
		sel = self._merge_sel(sel)
		return super().generate_selections(sel=sel, **kwargs)



class CachedView(SourceView, base.Container):
	def is_cached(self, item):
		return super(DataSource, self).__contains__(item)


	def __contains__(self, item):
		return self.is_cached(item) or (self.source is not None and item in self.source)


	def __getitem__(self, name):
		return super(DataSource, self).__getitem__(name)


	def __len__(self):
		return super(AbstractCountableData, self).__len__(item)


	def update(self, other): # TODO: maybe add a warning that dict.update is used
		return super(base.AbstractData, self).update(other)


	def get(self, name, default=None, **kwargs):
		if self.is_cached(name):
			return super(base.AbstractData, self).get(name, default)
		elif name in self:
			val = super().get(name, **kwargs)
			# if self.device is not None:
			# 	val = val.to(self.device)
			self[name] = val
			return self[name]
		return default


	def _find_missing(self, key, **kwargs):
		val = self.get(key, default=unspecified_argument, **kwargs)
		if val is unspecified_argument:
			return super()._find_missing(key)
		return val



Batchable.Batch = CachedView



class DataCollection(MultiModed, BufferTable, DataSource):
	def _title(self):
		mode = self.mode
		mode = '' if mode is None else f'<{mode}>'
		return f'{super()._title()}{mode}'


	def _update(self, sel=None, **kwargs):
		for name, buffer in self.iter_buffers(True):
			buffer.update(sel=sel, **kwargs)



class Subsetable(Epoched):
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


	def subset(self, cut=None, sel=None, shuffle=False, hard_copy=True):
		if sel is None:
			sel, _ = self._split_indices(indices=self.shuffle_indices(self.size(), gen=self.gen)
			if shuffle else torch.arange(self.size()), cut=cut)
		return self.create_view(sel=sel)


	def split(self, splits, shuffle=False):
		auto_name = isinstance(splits, (list, tuple, set))
		if auto_name:
			named_cuts = [(f'part{i}', r) for i, r in enumerate(splits)]
		else:
			assert isinstance(splits, dict), f'unknown splits: {splits}'
			assert not any(x for x in splits if x is None), 'names of splits cannot be None'
			named_cuts = list(splits.items())
		names, cuts = zip(*sorted(named_cuts, key=lambda nr: (isinstance(nr[1], int), isinstance(nr[1], float),
		                                                      nr[1] is None, nr[0]), reverse=True))

		remaining = self.size()
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
						rationums = [int(remaining * abs(ratio)) for ratio in ratios]
						nums.extend([int(math.copysign(1, r) * n) for r, n in zip(ratios, rationums)])
						remaining -= sum(rationums)
				if cut is None:
					pieces = len([cut, *itr])
					assert remaining > pieces, f'cant evenly distribute {remaining} samples into {pieces} cuts'
					evennums = [int(remaining // pieces) for _ in range(pieces)]
					nums.extend(evennums)
					remaining -= sum(evennums)

		if remaining > 0:
			nums[-1] += remaining

		indices = self.shuffle_indices(self.size(), gen=self.gen) if shuffle else torch.arange(self.size())

		plan = dict(zip(names, nums))
		parts = {}
		for name in sorted(names):
			num = plan[name]
			part, indices = self._split_indices(indices, num)
			parts[name] = self.subset(sel=part)
		if auto_name:
			return [parts[name] for name, _ in named_cuts]
		return parts



class Batch(Subsetable, CountableView, CachedView):
	pass



class View(Subsetable, CountableView, ReplacementView):
	pass



class Dataset(Subsetable, DataCollection):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._subset_src = None
	# 	# self._waiting_subset = None
	# 	# self._subset_indices = None


	View = View
	Batch = Batch


	# @property
	# def is_subset(self):
	# 	return self._subset_src is not None


	def _size(self):
		return next(iter(self.iter_buffers(True)))[1].size()


	def get_subset_src(self, recursive=True):
		if self._subset_src is None:
			return self
		return self._subset_src.get_subset_src(recursive=recursive) if recursive else self._subset_src


	@staticmethod
	def _create_buffer_view(buffer, sel=None, **kwargs):
		return buffer.create_view(sel=sel, **kwargs)


	def subset(self, cut=None, sel=None, shuffle=False, src_ref=True, hard_copy=True):
		if hard_copy:
			if sel is None:
				sel, _ = self._split_indices(indices=self.shuffle_indices(self.size(), gen=self.gen)
				if shuffle else torch.arange(self.size()), cut=cut)
			new = self.copy()
			if src_ref:
				new._subset_src = self
			new._default_len = len(sel)
			for name, buffer in self.buffers.items():
				new.register_buffer(name, buffer if isinstance(buffer, str)
				else self._create_buffer_view(buffer, sel=sel))
		else:
			new = super().subset(cut=cut, sel=sel, shuffle=shuffle)
		if self.mode is not None:
			self.register_modes(**{self.mode: new})
		return new

	
	def split(self, splits, shuffle=False, register_modes=False):
		parts = super().split(splits, shuffle=shuffle)
		if register_modes and isinstance(splits, dict):
			self.register_modes(**parts)
		return parts



class SimpleDataset(Dataset):
	_is_ready = True

	def __init__(self, data={}, **kwargs):
		super().__init__(**kwargs)
		for key, val in data.items():
			self.register_buffer(key, val)



class _ObservationInfo(DataSource):
	@property
	def din(self):
		return self.observation_space


	@property
	def observation_space(self):
		return self.space_of('observation')


	def get_observation(self, sel=None, **kwargs):
		return self.get('observation', sel=sel, **kwargs)



class ObservationDataset(_ObservationInfo, Dataset):
	class Batch(_ObservationInfo, Dataset.Batch):
		pass
	class View(_ObservationInfo, Dataset.View):
		pass


	# def _prepare(self, *args, **kwargs):
	# 	super()._prepare()
	# 	if not self.has_buffer('observation'):
	# 		# TODO: warning: guessing observation buffer
	# 		assert len(self.buffers), 'cant find a buffer for the observations (did you forget to register it?)'
	# 		key = list(self.buffers.keys())[-1]
	# 		self.register_buffer('observation', self.get_buffer(key))



class _SupervisionInfo(util.Metric, _ObservationInfo):
	@property
	def dout(self):
		return self.target_space


	@property
	def target_space(self):
		return self.space_of('target')


	def get_target(self, sel=None, **kwargs):
		return self.get('target', sel=sel, **kwargs)


	def difference(self, a, b, standardize=None):
		return self.dout.difference(a, b, standardize=standardize)


	def measure(self, a, b, standardize=None):
		return self.dout.measure(a, b, standardize=standardize)


	def distance(self, a, b, standardize=None):
		return self.dout.distance(a, b, standardize=standardize)



class SupervisedDataset(_SupervisionInfo, ObservationDataset):
	class Batch(_SupervisionInfo, ObservationDataset.Batch):
		pass
	class View(_SupervisionInfo, ObservationDataset.View):
		pass


	# def _prepare(self, *args, **kwargs):
	# 	super()._prepare()
	# 	if not self.has_buffer('target'):
	# 		# TODO: warning: guessing target buffer
	# 		key = list(self.buffers.keys())[0 if len(self.buffers) < 2 else -2]
	# 		self.register_buffer('target', self.get_buffer(key))



class _LabeledInfo(_SupervisionInfo):
	@property
	def label_space(self):
		return self.space_of('label')


	def get_label(self, sel=None, **kwargs):
		return self.get('label', sel=sel, **kwargs)



class LabeledDataset(_LabeledInfo, SupervisedDataset):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.register_buffer('target', 'label')


	class Batch(_LabeledInfo, SupervisedDataset.Batch):
		pass
	class View(_LabeledInfo, SupervisedDataset.View):
		pass


	# def _prepare(self, *args, **kwargs):
	# 	if not self.has_buffer('target') and self.has_buffer('label'):
	# 		self.register_buffer('target', self.get_buffer('label'))
	# 	super()._prepare(*args, **kwargs)
		# if not self.has_buffer('label'):
		# 	# TODO: warning: guessing target buffer
		# 	key = list(self.buffers.keys())[0 if len(self.buffers) < 3 else -3]
		# 	self.register_buffer('label', self.get_buffer(key))


	def generate_label(self, N, seed=None, gen=None):
		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen
		return self.label_space.sample(N, seed=seed, gen=gen)


	def generate_observation_from_label(self, label, seed=None, gen=None):
		raise NotImplementedError



# Labeled means there exists a deterministic mapping from labels to observations
# (not including possible subsequent additive noise)



class _SyntheticInfo(_LabeledInfo):
	_distinct_mechanisms = True


	@property
	def mechanism_space(self):
		return self.space_of('mechanism')


	def transform_to_mechanisms(self, data):
		if not self._distinct_mechanisms:
			return data
		return self.mechanism_space.transform(data, self.label_space)


	def transform_to_labels(self, data):
		if not self._distinct_mechanisms:
			return data
		return self.label_space.transform(data, self.mechanism_space)



class SyntheticDataset(_SyntheticInfo, LabeledDataset):
	_standardize_scale = True
	_use_mechanisms = True

	def __init__(self, standardize_scale=None, use_mechanisms=None, **kwargs):
		super().__init__(**kwargs)
		if standardize_scale is not None:
			self._standardize_scale = standardize_scale
		if use_mechanisms is not None:
			self._use_mechanisms = use_mechanisms
		self.register_buffer('mechanism', 'label')
		self.register_buffer('target', 'mechanism' if self._use_mechanisms else 'label')


	class Batch(_SyntheticInfo, LabeledDataset.Batch):
		@property
		def _distinct_mechanisms(self):
			return self.source._distince_mechanisms
	class View(_SyntheticInfo, LabeledDataset.View):
		@property
		def _distinct_mechanisms(self):
			return self.source._distince_mechanisms


	def generate_mechanism(self, N, seed=None, gen=None): # TODO: link with prior
		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen
		return self.mechanism_space.sample(N, gen=gen)


	def generate_observation_from_label(self, label, seed=None, gen=None):
		return self.generate_observation_from_mechanism(self.transform_to_mechanisms(label), seed=seed, gen=gen)


	def generate_observation_from_mechanism(self, mechanism, seed=None, gen=None):
		raise NotImplementedError
	# TODO: link with generative model
# Synthetic means the mapping is known (and available, usually only for evaluation)
# TODO: separate labels and mechanisms



class RootedDataset(DataCollection, Rooted):
	@classmethod
	def _infer_root(cls, root=None):
		return super()._infer_root(root=root) / 'datasets'


	@agnosticmethod
	def get_root(self, dataset_dir=None):
		if dataset_dir is None:
			dataset_dir = self.name
		root = super().get_root() / dataset_dir
		os.makedirs(str(root), exist_ok=True)
		return root


	def get_aux_root(self, dataset_dir=None):
		root = self.get_root(dataset_dir=dataset_dir) / 'aux'
		os.makedirs(str(root), exist_ok=True)
		return root
	
	
	def _find_path(self, dataset_name='', file_name=None, root=None):
		if root is None:
			root = self.root
		*other, dataset_name = dataset_name.split('.')
		if file_name is None:
			file_name = '.'.join(other) if len(other) else self.name
		path = root / f'{file_name}.h5'
		return path, dataset_name


	_default_hdf_buffer_type = HDFBuffer
	def register_hdf_buffer(self, name, dataset_name, file_name=None, root=None,
	                        buffer_type=None, path=None, **kwargs):
		if buffer_type is None:
			buffer_type = self._default_hdf_buffer_type
		if path is None:
			path, dataset_name = self._find_path(dataset_name, file_name=file_name, root=root)
		return self.register_buffer(name, buffer_type=buffer_type, dataset_name=dataset_name, path=path, **kwargs)


	@staticmethod
	def create_hdf_dataset(path, dataset_name, data=None, meta=None, dtype=None, shape=None):
		# if file_name is unspecified_argument:
		# 	file_name = 'aux'
		# if path is None:
		# 	path, dataset_name = self._find_path(dataset_name, file_name=file_name, root=root)
		
		if isinstance(data, torch.Tensor):
			data = data.detach().cpu().numpy()
		with hf.File(path, 'a') as f:
			if data is not None or (dtype is not None and shape is not None):
				f.create_dataset(dataset_name, data=data, dtype=dtype, shape=shape)
			if meta is not None:
				f.attrs[dataset_name] = json.dumps(meta, sort_keys=True)
		return path, dataset_name



class EncodableDataset(ObservationDataset, RootedDataset):
	def __init__(self, encoder=None, replace_observation_key=None, encoded_key='encoded',
	             encoded_file_name='aux', encode_on_load=False, save_encoded=False, encode_pbar=None, **kwargs):
		super().__init__(**kwargs)
		self._replace_observation_key = replace_observation_key
		self._encoded_observation_key = encoded_key
		self._encoded_file_name = encoded_file_name
		self._encode_on_load = encode_on_load
		self._save_encoded = save_encoded
		self._encode_pbar = encode_pbar
		self.encoder = encoder
	

	@property
	def encoder(self):
		return self._encoder
	@encoder.setter
	def encoder(self, encoder):
		buffer = self.get_buffer(self._encoded_observation_key)
		if buffer is not None:
			buffer.encoder = encoder
		self._encoder = encoder
	
	
	def _get_code_path(self, file_name='aux', root=None):
		return None if file_name is None else self._find_path(file_name=file_name, root=root)[0]
	
	
	@staticmethod
	def _encoder_save_key(encoder):
		info = encoder.get_encoder_fingerprint()
		ident = md5(json.dumps(info, sort_keys=True))
		return ident, info
		
	
	def load_encoded_data(self, encoder=None, source_key='observation',
	                      batch_size=None, save_encoded=None,
	                      file_name=unspecified_argument, root=None):
		if encoder is None:
			encoder = self.encoder
		if file_name is unspecified_argument:
			file_name = self._encoded_file_name
		if save_encoded is None:
			save_encoded = self._save_encoded
		data = None
		
		path = self._get_code_path(file_name=file_name, root=root)
		if path is not None and path.exists() and encoder is not None:
			ident, _ = self._encoder_save_key(encoder)
			with hf.File(str(path), 'r') as f:
				if ident in f:
					data = f[ident][()]
					data = torch.from_numpy(data)
		
		if data is None and self._encode_on_load:
			batches = []
			for batch in self.get_iterator(batch_size=batch_size, shuffle=False, force_batch_size=False,
			                               pbar=self._encode_pbar):
				with torch.no_grad():
					batches.append(encoder(batch[source_key]))
			
			data = torch.cat(batches)
			if save_encoded:
				self.save_encoded_data(encoder, data, path)
		
		return data
	
	
	@classmethod
	def save_encoded_data(cls, encoder, data, path):
		ident, info = cls._encoder_save_key(encoder)
		cls.create_hdf_dataset(path, ident, data=data, meta=info)
		
	
	def _prepare(self, *args, **kwargs):
		super()._prepare(*args, **kwargs)
		
		if self._replace_observation_key is not None:
			self._encoded_observation_key = 'observation'
			self.register_buffer(self._replace_observation_key, self.get_buffer('observation'))
		
		self.register_buffer(self._encoded_observation_key,
		                     buffer=self.EncodedBuffer(encoder=self.encoder, source=self.get_buffer('observation'),
		                                        data=self.load_encoded_data()))
	
	
	class EncodedBuffer(BufferView):
		def __init__(self, encoder=None, max_batch_size=64, **kwargs):
			super().__init__(**kwargs)
			# if encoder is not None and encoder_device is None:
			# 	encoder_device = getattr(encoder, 'device', None)
			# self._encoder_device = encoder_device
			self.encoder = encoder
			self.max_batch_size = max_batch_size
			

		@property
		def encoder(self):
			return self._encoder
		@encoder.setter
		def encoder(self, encoder):
			self._encoder = encoder
			if encoder is not None and hasattr(encoder, 'dout'):
				self.space = getattr(encoder, 'dout', None)
			# self._encoder_device = getattr(encoder, 'device', self._encoder_device)


		def _encode_raw_observations(self, observations):
			# device = observations.device
			if len(observations) > self.max_batch_size:
				samples = []
				batches = observations.chunk(self.max_batch_size)
				for batch in batches:
					# with torch.no_grad():
					# if self._encoder_device is not None:
					# 	batch = batch.to(self._encoder_device)
					samples.append(self.encoder.encode(batch))#.to(device))
				return torch.cat(samples)
			# with torch.no_grad():
			# if self._encoder_device is not None:
			# 	observations = observations.to(self._encoder_device)
			return self.encoder.encode(observations)#.to(device)


		def _get(self, sel=None, **kwargs):
			sample = super()._get(sel=sel, **kwargs)
			if self.data is None and self.encoder is not None:
				sample = self._encode_raw_observations(sample)
			if sel is None:
				self.data = sample
			return sample



class ImageDataset(ObservationDataset, RootedDataset):
	def __init__(self, download=False, **kwargs):
		super().__init__(**kwargs)
		self._auto_download = download


	@classmethod
	def download(cls, **kwargs):
		raise NotImplementedError


	class DatasetNotDownloaded(FileNotFoundError):
		def __init__(self):
			super().__init__('use download=True to enable automatic download.')


	# @classmethod
	# def _default_buffer_factory(cls, ):
	#
	#
	# 	pass










