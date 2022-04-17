import inspect

from omnibelt import get_printer, unspecified_argument, agnosticmethod, classdescriptor, ClassDescriptable

from .util import spaces

prt = get_printer(__file__)



class Hyperparameter(classdescriptor, property):
	def __init__(self, name=None, fget=None, default=None, required=False,
	             strict=False, cache=False, space=None, **kwargs):
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
		self.strict = strict # raises and error if an invalid value is set


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


	def reset(self):
		self.value = self._missing


	def __str__(self):
		try:
			value = self.get_value()
		except self.MissingHyperparameter:
			value = '?'
		return f'{self.__class__}({value})'


	def __get__(self, obj, cls=None):
		return self.get_value(obj, cls=cls)


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
		if self.required:
			raise self.MissingHyperparameter(self.name)
		return self.default


	def validate_value(self, value):
		if self.space is not None:
			self.space.validate(value)


	def update_value(self, value, obj=None):
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


	def __set__(self, obj, value):
		self.update_value(value, obj=obj)
		return self



class Parametrized(metaclass=ClassDescriptable):
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
		def __init__(self, default=None, required=True, module=None, cache=False, **kwargs):
			super().__init__(default=default, required=required, cache=cache or module is not None, **kwargs)
			self.module = module


		class InvalidInstance(Hyperparameter.InvalidValue):
			def __init__(self, name, value, base, msg=None):
				if msg is None:
					msg = f'{name}: {value} (should be an instance of {base})'
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
		self.name = name
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
		self.kwargs['fget'] = self.fn
		self.name = name
		setattr(obj, name, obj.register_hparam(name, **self.kwargs))

	
	def getter(self, fn):
		self.fget = fn
		return self


	def setter(self, fn):
		self.fset = fn
		return self


	def deleter(self, fn):
		self.fdel = fn
		return self




# class Property(object):
#     "Emulate PyProperty_Type() in Objects/descrobject.c"
#
#     def __init__(self, fget=None, fset=None, fdel=None, doc=None):
#         self.fget = fget
#         self.fset = fset
#         self.fdel = fdel
#         if doc is None and fget is not None:
#             doc = fget.__doc__
#         self.__doc__ = doc
#
#     def __get__(self, obj, objtype=None):
#         if obj is None:
#             return self
#         if self.fget is None:
#             raise AttributeError("unreadable attribute")
#         return self.fget(obj)
#
#     def __set__(self, obj, value):
#         if self.fset is None:
#             raise AttributeError("can't set attribute")
#         self.fset(obj, value)
#
#     def __delete__(self, obj):
#         if self.fdel is None:
#             raise AttributeError("can't delete attribute")
#         self.fdel(obj)
#
#     def getter(self, fget):
#         return type(self)(fget, self.fset, self.fdel, self.__doc__)
#
#     def setter(self, fset):
#         return type(self)(self.fget, fset, self.fdel, self.__doc__)
#
#     def deleter(self, fdel):
#         return type(self)(self.fget, self.fset, fdel, self.__doc__)



