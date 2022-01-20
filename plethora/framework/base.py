
from omnibelt import unspecified_argument


class Buffer:
	space = None
	
	def __init__(self, data=None, space=unspecified_argument, **kwargs):
		super().__init__(**kwargs)
		self.data = data
		if space is unspecified_argument:
			space = self.space
		self.space = space



class Function:
	din, dout = None, None
	
	def __init__(self, *args, din=unspecified_argument, dout=unspecified_argument, **kwargs):
		super().__init__(*args, **kwargs)
		if din is unspecified_argument:
			din = self.din
		if dout is unspecified_argument:
			dout = self.dout
		self.din, self.dout = din, dout
	
	
	def get_dims(self):
		return self.din, self.dout
	










