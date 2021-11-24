



class Dimension:
	din, dout = None, None

	def __init__(self, *args, din=None, dout=None, **kwargs):
		super().__init__(*args, **kwargs)
		if din is None:
			din = self.din
		if dout is None:
			dout = self.dout
		self.din, self.dout = din, dout


	def get_dims(self):
		return self.din, self.dout



class Device:
	def __init__(self, *args, device=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.device = device




