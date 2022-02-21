

from ...framework import util



class PathDiscriminator:
	def __init__(self, discriminator, **kwargs):
		super().__init__(**kwargs)
		self.discriminator = discriminator


	def judge(self, paths):
		B, K, *_ = paths.shape
		samples = util.combine_dims(paths, 0, 2)
		scores = self.discriminator(samples)
		return util.split_dim(scores, B, K)

