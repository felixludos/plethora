import inspect

import logging
from omnibelt import get_printer, unspecified_argument, agnosticmethod, classdescriptor, ClassDescriptable

from .util import spaces

# prt = get_printer(__file__, format='%(levelname)s: %(msg)s')

prt = logging.Logger('Hyperparameters')
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(levelname)s: %(msg)s'))
ch.setLevel(0)
prt.addHandler(ch)



class Hyperparameter(property, classdescriptor):
	def __init__(self, name=None, fget=None, default=unspecified_argument, required=False,
	             strict=False, cache=False, fixed=False, space=None, **kwargs):
		super().__init__(fget=fget, **kwargs)
		if name is None:
			assert fget is not None, 'No name provided'
			name = fget.__name__
		self.name = name
		self.value = self._missing
		self.default = default
		self.cache = cache
		if space is not None and isinstance(space, (list, tuple, set)):
			space = spaces.CategoricalDim(space)
		self.space = space
		self.required = required
		self.fixed = fixed
		self.strict = strict # raises and error if an invalid value is set


	def getter(self, fn):
		self.fget = fn
		return self


	def setter(self, fn):
		self.fset = fn
		return self


	def deleter(self, fn):
		self.fdel = fn
		return self


	def copy(self):
		raise NotImplementedError # TODO


	class _missing(Exception):
		pass


	class MissingHyperparameter(KeyError):
		pass


	class InvalidValue(Exception):
		def __init__(self, name, value, msg=None):
			if msg is None:
				msg = f'{name}: {value}'
			super().__init__(msg)
			self.name = name
			self.value = value


	def reset(self, obj=None):
		self.value = self._missing
		if self.fdel is not None and obj is not None:
			super().__delete__(obj)


	def __str__(self):
		try:
			value = self.get_value()
		except self.MissingHyperparameter:
			value = '?'
		return f'{self.__class__}({value})'


	def __get__(self, obj, cls=None):
		return self.get_value(obj, cls=cls)


	def __delete__(self, obj): # TODO: test this
		self.reset()


	def __set__(self, obj, value):
		self.update_value(value, obj=obj)
		return self


	def get_value(self, obj=None, cls=None):
		if self.value is not self._missing:
			return self.value
		if obj is not None:
			try:
				value = super().__get__(obj, cls)
			except Hyperparameter.MissingHyperparameter:
				raise self.MissingHyperparameter(self.name)
			if self.cache:
				self.value = value
			return value
		if self.required or self.default is unspecified_argument:
			raise self.MissingHyperparameter(self.name)
		return self.default


	class FixedHyperparameter(Exception):
		def __init__(self, msg='cant change'):
			super().__init__(msg)


	def validate_value(self, value):
		if self.space is not None:
			try:
				self.space.validate(value)
			except self.space.InvalidValue:
				raise self.InvalidValue


	def update_value(self, value, obj=None):
		if self.fixed:
			raise self.FixedHyperparameter
		try:
			self.validate_value(value)
		except self.InvalidValue as e:
			if self.strict:
				raise e
			prt.warning(f'{type(e).__name__}: {e}')
		if self.fset is not None and obj is not None:
			self.reset()
			return super().__set__(obj, value)
		else:
			self.value = value
		return value



class Parametrized(metaclass=ClassDescriptable):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **self._extract_hparams(kwargs))


	def _extract_hparams(self, kwargs):
		for key in self.iterate_hparams():
			if key in kwargs:
				setattr(self, key, kwargs[key])
				del kwargs[key]
		return kwargs


	Hyperparameter = Hyperparameter
	@agnosticmethod
	def register_hparam(self, name=None, fget=None, default=unspecified_argument, required=False, **kwargs):
		return self.Hyperparameter(fget=fget, required=required, default=default, name=name, **kwargs)


	RequiredHyperparameter = Hyperparameter.MissingHyperparameter


	@agnosticmethod
	def reset_hparams(self):
		for key, param in self.iterate_hparams(True):
			param.reset()


	@agnosticmethod
	def iterate_hparams(self, items=False, **kwargs):
		for key, val in self.__dict__.items():
			if isinstance(val, Hyperparameter):
				yield (key, val) if items else key
		if not isinstance(self, type):
			yield from self.__class__.iterate_hparams(items=items, **kwargs)


	@agnosticmethod
	def inherit_hparams(self, *names, copy=False, strict=False):
		for name in names:
			hparam = inspect.getattr_static(self, name, unspecified_argument)
			if hparam is unspecified_argument:
				if strict:
					raise self.Hyperparameter.MissingHyperparameter(name)
			else:
				if copy:
					hparam = hparam.copy()
				setattr(self, name, hparam)



class ModuleParametrized(Parametrized):
	class Hyperparameter(Parametrized.Hyperparameter):
		def __init__(self, default=unspecified_argument, required=True, module=None, cache=False, **kwargs):
			super().__init__(default=default, required=required, cache=cache or module is not None, **kwargs)
			self.module = module


		class InvalidInstance(Hyperparameter.InvalidValue):
			def __init__(self, name, value, base, msg=None):
				if msg is None:
					msg = f'{name}: {value} (expecting an instance of {base})'
				super().__init__(name, value, msg=msg)
				self.base = base


		def validate_value(self, value):
			if self.module is not None and not isinstance(value, self.module):
				raise self.InvalidInstance(self.name, value, self.module)



class inherit_hparams:
	def __init__(self, *names, copy_hparam=False, strict=False, **kwargs):
		self.names = names
		self.copy_hparams = copy_hparam
		self.strict = strict
		self.kwargs = kwargs


	def __call__(self, cls):
		cls.inherit_hparams(*self.names, copy_hparms=self.copy_hparams, strict=self.strict, **self.kwargs)
		return cls



class hparam:
	def __init__(self, default=unspecified_argument, space=None, name=None, **kwargs):
		self.default = default
		assert name is None, 'Cannot specify a different name with hparam'
		self.name = None
		self.space = space
		self.kwargs = kwargs


	def __call__(self, fn):
		self.fn = fn


	def __get__(self, instance, owner): # TODO: this is just for linting, right?
		return getattr(instance, self.name)


	def __set_name__(self, obj, name):
		if self.default is not unspecified_argument:
			self.kwargs['default'] = self.default
		self.kwargs['space'] = self.space
		self.kwargs['fget'] = getattr(self, 'fn', None)
		self.name = name
		setattr(obj, name, obj.register_hparam(name, **self.kwargs))






