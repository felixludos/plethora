from omnibelt import get_printer, unspecified_argument, agnosticmethod

prt = get_printer(__file__)



class Hyperparameter(property):
	def __init__(self, name=None, fget=None, default=None, required=False,
	             strict=False, cache=False, **kwargs):
		super().__init__(fget=fget, **kwargs)
		if name is None:
			assert fget is not None, 'No name provided'
			name = fget.__name__
		self.name = name
		self.value = self._missing
		self.default = default
		self.cache = cache
		self.required = required
		self.strict = strict # raises and error if an invalid value is set


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
			value = super().__get__(obj, cls)
			if self.cache:
				self.value = value
			return value
		if self.required:
			raise self.MissingHyperparameter(self.name)
		return self.default


	def validate_value(self, value):
		pass


	def update_value(self, value, obj=None):
		try:
			self.validate_value(value)
		except self.InvalidValue as e:
			if self.strict:
				raise e
			prt.warning(f'{type(e).__name__}: {e}')
		if self.fset is not None and obj is not None:
			return super().__set__(obj, value)
		else:
			self.value = value
		return value


	def __set__(self, obj, value):
		self.update_value(value, obj=obj)
		return self



class Parametrized:
	Hyperparameter = Hyperparameter
	@agnosticmethod
	def register_hparam(self, name=None, fget=None, default=unspecified_argument, required=False):
		return self.Hyperparameter(fget=fget, required=required, default=default, name=name)


	@agnosticmethod
	def iterate_hparams(self, items=False, **kwargs):
		for key, val in self.__dict__.items():
			if isinstance(val, Hyperparameter):
				yield (key, val) if items else key
		if not isinstance(self, type):
			yield from self.__class__.iterate_hparams(items=items, **kwargs)


	@agnosticmethod
	def inherit_hparams(self, *names, copy_hparams=False, strict=False):
		for name in names:
			hparam = getattr(self, name, unspecified_argument)
			if hparam is unspecified_argument:
				if strict:
					raise self.Hyperparameter.MissingHyperparameter(name)
			else:
				if copy_hparams:
					hparam = hparam.copy()
				setattr(self, name, hparam)



class ModuleParametrized(Parametrized):
	class Hyperparameter(Parametrized.Hyperparameter):
		def __init__(self, default=None, required=True, module=None, **kwargs):
			super().__init__(default=default, required=required, **kwargs)
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
	def __init__(self, **kwargs):
		self.kwargs = kwargs


	def __call__(self, fn):
		self.fn = fn


	def __set_name__(self, obj, name):
		self.kwargs['fget'] = self.fn
		setattr(obj, name, obj.register_hparam(**self.kwargs))




