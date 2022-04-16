import inspect
from omnibelt import auto_args



class data_args(auto_args):
	def __init__(self, _out_key=None, _show_info=True, **aliases):
		super().__init__()
		self.out_key = _out_key
		self.show_info = _show_info
		self.aliases = aliases


	def process_args(self, *args, info=None, **kwargs):
		self.info = info
		if info is None:
			return args, kwargs
		for key, alias in self.aliases.items():
			if key not in kwargs and key in info:
				kwargs[key] = info[alias]

		params = inspect.signature(self.fn).parameters
		if 'info' in params:
			kwargs['info'] = info
		return (), kwargs


	def process_out(self, out):
		if self.out_key is None or self.info is None:
			return out
		info = self.info
		info[self.out_key] = out
		del self.info # dont hold on to info after call is complete
		return info




















