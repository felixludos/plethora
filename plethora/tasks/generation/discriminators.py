




class DistributionDiscriminator:
	def __init__(self, dataset=None, extractor=None, criterion=None, **kwargs):
		super().__init__(**kwargs)
		self.dataset = dataset
		self.extractor = extractor
		self.criterion = criterion


	def judge(self, samples):
		return self.extractor.extract(samples)


	def compare(self, features, dataset):
		raise NotImplementedError









