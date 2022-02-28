import math
from collections import OrderedDict
import torch
from omnibelt import unspecified_argument, duplicate_instance

from ..framework import base, Fileable
from .buffers import TensorBuffer, WrappedBuffer

class MissingModeError(Exception):
	pass



class BufferNotFoundError(Exception):
	pass



class Iterable:
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


	def create_batch(self, sel=None, **kwargs):
		raise NotImplementedError
	


class Batchable(Iterable, base.Seeded):
	def __init__(self, batch_size=64, shuffle_batches=True, force_batch_size=True,
	             batch_device=None, infinite=False, **kwargs):
		super().__init__(**kwargs)

		self._batch_size = batch_size
		self._force_batch_size = force_batch_size
		self._shuffle_batches = shuffle_batches
		self._batch_device = batch_device
		self._infinite = infinite


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


	def generate_selections(self, sel=None, num_samples=None, batch_size=None, shuffle=False, force_batch_size=None, gen=None, **kwargs):
		if batch_size is None:
			batch_size = self.batch_size
		if force_batch_size is None:
			force_batch_size = self._force_batch_size
		if gen is None:
			gen = self.gen
			
		if sel is None:
			sel = torch.arange(len(self))
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

		N = len(self) if sel is None else len(sel)
		samples_per_epoch = N - int(force_batch_size) * (N % batch_size)
		batches_per_epoch = int(math.ceil(samples_per_epoch / batch_size))
		if infinite is None:
			total_samples = None
		elif num_batches is not None:
			total_samples = (num_batches % batches_per_epoch) * batch_size \
			                + (num_batches // batches_per_epoch) * samples_per_epoch
		elif num_samples is not None:
			total_samples = num_samples - int(force_batch_size or hard_limit) * (num_samples % batch_size) \
			                + int(not hard_limit and num_samples % batch_size > 0) * batch_size
		else:
			total_samples = samples_per_epoch * epochs
		if pbar is not None:
			pbar = pbar(total=total_samples)

		while total_samples is None or total_samples > 0:
			sels = self.generate_selections(sel=sel, num_samples=total_samples, batch_size=batch_size, shuffle=shuffle,
			                                force_batch_size=force_batch_size, gen=gen, **kwargs)
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



class Batch(Batchable, base.Container):
	def __init__(self, source, sel=None, **kwargs):
		super().__init__(**kwargs)
		self.source = source
		self.sel = sel
	
	
	def get_iterator(self, *, sel=None, **kwargs):
		if sel is None:
			sel = self.sel
		return super().get_iterator(sel=sel, **kwargs)


	def generate_selections(self, *, sel=None, **kwargs):
		if sel is None:
			sel = self.sel
		return super().generate_selections(sel=sel, **kwargs)


	def get_available(self):
		return list(self.source.buffers.keys())


	def get_space(self, name):
		return self.source.get_space(name)


	def _find_missing(self, key, **kwargs):
		val = self.source.get(key, self.sel, **kwargs)
		if self.device is not None:
			val = val.to(self.device)
		self[key] = val # cache the results
		return val


	@property
	def size(self):
		return len(self.sel)


	def create_batch(self, batch_type=None, **kwargs):
		if batch_type is None:
			batch_type = self.__class__
		return batch_type(self.source, **kwargs)


	def __str__(self):
		entries = list(self.keys())
		for available in self.get_available():
			if available not in entries:
				entries.append('{' + available + '}')
		entries = ', '.join(entries)
		return f'{self.__class__.__name__}[{self.size}]({entries})'



class DataCollection(base.Buffer, Iterable):
	name = None
	
	def __init__(self, *, name=None, mode=None,
	             buffers=None, modes=None, space=None, **kwargs):
		super().__init__(space=None, **kwargs)
		if self.name is None or name is not None:
			self.name = name

		# if batch_device is unspecified_argument:
		# 	batch_device = None
		# self.batch_device = batch_device

		if buffers is None:
			buffers = OrderedDict()
		self.buffers = buffers

		self._mode = mode
		if modes is None:
			modes = {}
			if mode is not None:
				modes[mode] = self
		self._modes = modes


	def get_name(self):
		return self.__class__.__name__ if self.name is None else self.name


	def __str__(self):
		# return self.get_name()
		name = '' #if self.name is None else '{' + self.name + '}'
		return f'{self.__class__.__name__}{name}<{self.mode}>'


	def __repr__(self):
		return str(self)
		# name = '' if self.name is None else self.name
		# return f'{self.__class__.__name__}({name})'


	def copy(self):
		new = super().copy()
		new.buffers = new.buffers.copy()
		new._modes = new._modes.copy()
		return new


	_default_buffer_type = None
	@classmethod
	def _default_buffer_factory(cls, name, buffer_type=None, **kwargs):
		if buffer_type is None:
			buffer_type = cls._default_buffer_type
		return buffer_type(name, **kwargs)


	def register_buffer(self, name, buffer=None, space=unspecified_argument, buffer_type=None, **kwargs):
		if not isinstance(buffer, base.Buffer):
			if buffer_type is None and type(buffer) == type and issubclass(buffer, base.Buffer):
				buffer_type = buffer
			if isinstance(buffer, torch.Tensor):
				kwargs['data'] = buffer
			buffer = self._default_buffer_factory(name, buffer_type=buffer_type, **kwargs)
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


	def _package_sample(self, sample, name, sel, **kwargs):
		return sample


	def get(self, name, sel=None, **kwargs):
		return super().get(sel=sel, name=name, **kwargs)


	def _get(self, name, sel=None, **kwargs):
		self._check_buffer_names(name)
		sample = self.buffers[name].get(sel)
		return self._package_sample(sample, name, sel, **kwargs)


	def register_modes(self, **modes):
		for name, mode in modes.items():
			mode._mode = name
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


	
class Dataset(DataCollection, Batchable, base.FixedBuffer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._subset_src = None
		self._waiting_subset = None
		self._subset_indices = None


	def __str__(self):
		return f'{self.__class__.__name__}<{self.mode}>[{len(self)}]'


	_default_buffer_type = TensorBuffer


	_default_batch_type = Batch
	def create_batch(self, batch_type=None, **kwargs):
		if batch_type is None:
			batch_type = self._default_batch_type
		return batch_type(self, **kwargs)


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


	_default_wrapper_type = WrappedBuffer
	@classmethod
	def _wrap_buffer(cls, source, sel=None, wrapper_type=None, **kwargs):
		if wrapper_type is None:
			wrapper_type = cls._default_wrapper_type
		return wrapper_type(source, sel=sel, **kwargs)


	@property
	def is_subset(self):
		return self._subset_src is not None


	def get_subset_src(self, recursive=True):
		if self._subset_src is None:
			return self
		return self._subset_src.get_subset_src(recursive=recursive) if recursive else self._subset_src


	def subset(self, cut=None, indices=None, shuffle=False, src_ref=True):
		if indices is None:
			indices, _ = self._split_indices(indices=self.shuffle_indices(len(self), gen=self.gen)
										  if shuffle else torch.arange(len(self)), cut=cut)
		new = self.copy()
		if src_ref:
			new._subset_src = self
		new._default_len = len(indices)
		for name, buffer in self.buffers.items():
			new.register_buffer(name, self._wrap_buffer(buffer, sel=indices))
		if self.mode is not None:
			self.register_modes(**{self.mode: new})
		return new


	def _count(self):
		return len(next(iter(self.buffers.values())))
	
	
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
						rationums = [int(remaining*abs(ratio)) for ratio in ratios]
						nums.extend([int(math.copysign(1, r)*n) for r, n in zip(ratios, rationums)])
						remaining -= sum(rationums)
				if cut is None:
					pieces = len([cut, *itr])
					assert remaining > pieces, f'cant evenly distribute {remaining} samples into {pieces} cuts'
					evennums = [int(remaining//pieces) for _ in range(pieces)]
					nums.extend(evennums)
					remaining -= sum(evennums)

		if remaining > 0:
			nums[-1] += remaining

		indices = self.shuffle_indices(len(self), gen=self.gen) if shuffle else torch.arange(len(self))

		plan = dict(zip(names, nums))
		parts = {}
		for name in sorted(names):
			num = plan[name]
			part, indices = self._split_indices(indices, num)
			parts[name] = self.subset(indices=part)

		if register_modes:
			self.register_modes(**parts)

		if auto_name:
			return [parts[name] for name, _ in named_cuts]
		return parts



class ObservationDataset(Dataset):
	@property
	def din(self):
		return self.get_observation_space()


	def get_observation_space(self):
		return self.get_space('observation')


	def get_observation(self, sel=None, **kwargs):
		return self.get('observation', sel=sel, **kwargs)


	def _load(self, *args, **kwargs):
		super()._load()
		if not self.has_buffer('observation'):
			# TODO: warning: guessing observation buffer
			assert len(self.buffers), 'cant find a buffer for the observations (did you forget to register it?)'
			key = list(self.buffers.keys())[-1]
			self.register_buffer('observation', self._wrap_buffer(self.get_buffer(key)))


	# def __len__(self):
	# 	return len(self.get_buffer('observation'))



class SupervisedDataset(ObservationDataset):
	@property
	def dout(self):
		return self.get_target_space()


	def get_target_space(self):
		return self.get_space('target')


	def get_target(self, sel=None, **kwargs):
		return self.get('target', sel=sel, **kwargs)


	def _load(self, *args, **kwargs):
		super()._load()
		if not self.has_buffer('target'):
			# TODO: warning: guessing target buffer
			key = list(self.buffers.keys())[0 if len(self.buffers) < 2 else -2]
			self.register_buffer('target', self._wrap_buffer(self.get_buffer(key)))



class LabeledDataset(SupervisedDataset):
	def get_label_space(self):
		return self.get_space('label')


	def get_label(self, sel=None, **kwargs):
		return self.get('label', sel=sel, **kwargs)


	def _load(self, *args, **kwargs):
		if not self.has_buffer('target') and self.has_buffer('label'):
			self.register_buffer('target', self._wrap_buffer(self.get_buffer('label')))
		super()._load()
		if not self.has_buffer('label'):
			# TODO: warning: guessing target buffer
			key = list(self.buffers.keys())[0 if len(self.buffers) < 3 else -3]
			self.register_buffer('label', self._wrap_buffer(self.get_buffer(key)))


	def generate_label(self, N, seed=None, gen=None):
		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen
		return self.get_label_space().sample(N, seed=seed, gen=gen)


	def generate_observation_from_label(self, label, seed=None, gen=None):
		raise NotImplementedError

# Labeled means there exists a deterministic mapping from labels to observations
# (not including possible subsequent additive noise)



class SyntheticDataset(LabeledDataset):
	def __init__(self, distinct_mechanisms=False, standardize_scale=True, **kwargs):
		super().__init__(**kwargs)
		self._distinct_mechanisms = distinct_mechanisms
		self._standardize_scale = standardize_scale


	def get_mechanism_space(self):
		return self.get_space('mechanism') if self._distinct_mechanisms else self.get_label_space()


	def _load(self, *args, **kwargs):
		if not self._distinct_mechanisms and not self.has_buffer('label') and self.has_buffer('mechanism'):
			self.register_buffer('label', self._wrap_buffer(self.get_buffer('mechanism')))
		super()._load()
		if not self.has_buffer('mechanism'):
			# TODO: warning: guessing target buffer
			key = list(self.buffers.keys())[0 if len(self.buffers) < 4 else -4] \
				if self._distinct_mechanisms else 'label'
			self.register_buffer('mechanism', self._wrap_buffer(self.get_buffer(key)))


	def transform_to_mechanisms(self, data):
		if not self._distinct_mechanisms:
			return data
		return self.get_mechanism_space().transform(data, self.get_label_space())


	def transform_to_labels(self, data):
		if not self._distinct_mechanisms:
			return data
		return self.get_label_space().transform(data, self.get_mechanism_space())


	def difference(self, a, b, standardize=None): # TODO: link to metric space
		if standardize is None:
			standardize = self._standardize_scale
		return self.get_mechanism_space().difference(a, b, standardize=standardize)


	def distance(self, a, b, standardize=None):  # TODO: link to metric space
		if standardize is None:
			standardize = self._standardize_scale
		return self.get_mechanism_space().distance(a,b, standardize=standardize)


	def generate_mechanism(self, N, seed=None, gen=None): # TODO: link with prior
		if seed is not None:
			gen = torch.Generator().manual_seed(seed)
		if gen is None:
			gen = self.gen
		return self.get_mechanism_space().sample(N, gen=gen)


	def generate_observation_from_label(self, label, seed=None, gen=None):
		return self.generate_observation_from_mechanism(self.transform_to_mechanisms(label), seed=seed, gen=gen)


	def generate_observation_from_mechanism(self, mechanism, seed=None, gen=None):
		raise NotImplementedError
	# TODO: link with generative model
# Synthetic means the mapping is known (and available, usually only for evaluation)
# TODO: separate labels and mechanisms



class SourcedDataset(DataCollection, Fileable):
	@classmethod
	def _infer_root(cls, root=None):
		return super()._infer_root(root=root) / 'datasets'


	def get_root(self):
		root = super().get_root() / self.get_name()
		os.makedirs(str(root), exist_ok=True)
		return root


	_default_hdf_buffer_type = None
	def register_hdf_buffer(self, name, dataset_name, file_name=None, root=None, **kwargs):
		if root is None:
			root = self.get_root()

		*other, dataset_name = dataset_name.split('.')
		if file_name is None:
			file_name = '.'.join(other) if len(other) else 'aux'

		path = root / f'{file_name}.h5'
		# return self.register_buffer(name, buffer_type= path=path)



	# def register_buffer(self, name, buffer=None, space=unspecified_argument, buffer_type=None, **kwargs):
	# 	if not isinstance(buffer, base.Buffer):
	# 		if buffer_type is None and issubclass(buffer, base.Buffer):
	# 			buffer_type = buffer
	# 		elif 'data' not in kwargs and buffer is not None:
	# 			kwargs['data'] = buffer
	# 		buffer = self._default_buffer_factory(name, buffer_type=buffer_type, **kwargs)
	# 	if space is not unspecified_argument:
	# 		buffer.set_space(space)
	# 	self.buffers[name] = buffer
	# 	return self.buffers[name]





class ImageDataset(ObservationDataset, SourcedDataset):
	# def __init__(self, **kwargs):
	# 	pass

	@classmethod
	def download(cls, **kwargs):
		raise NotImplementedError


	@classmethod
	def _default_buffer_factory(cls, ):


		pass






