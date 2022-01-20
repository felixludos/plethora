
class Estimator:
	'''
	Can predict a label from observations
	'''
	def predict(self, observation: 'OBSERVATION') -> 'LABEL':
		'''
		Predicts the labels corresponding to the given observation
		:param observation: input
		:return: output prediction
		'''
		raise NotImplementedError


class Extractor:
	'''
	Can extract features from observations
	'''
	def extract(self, observation: 'OBSERVATION') -> 'FEATURES':
		'''
		Extracts features from the given observation
		:param observation: input
		:return: corresponding features
		'''
		raise NotImplementedError


class Encoder(Extractor):
	'''
	Can encode observations into latent codes (should be reversible)
	'''
	def encode(self, observation: 'OBSERVATION') -> 'LATENT':
		'''
		Encodes the observations into latent codes
		:param observation: observations that should be encoded
		:return: corresponding latent codes
		'''
		raise NotImplementedError
	
	
	def extract(self, observation: 'OBSERVATION') -> 'FEATURES':
		return self.encode(observation)
	

class Decoder:
	'''
	Can decode latent codes to recover the corresponding observations
	'''
	def decode(self, latent: 'LATENT') -> 'OBSERVATION':
		'''
		Decodes the latent code to recover observations
		:param latent: code
		:return: observations corresponding to the latent code
		'''
		raise NotImplementedError
	

class Generator:
	'''
	Can generate arbitrarily many observations from random bits
	'''
	def generate(self, N: int, seed: int = None) -> 'OBSERVATION':
		'''
		Generates new observations (from random bits)
		:param N: number of samples to generate
		:param seed: for generator
		:return: N generated observations
		'''
		raise NotImplementedError
	

class Criterion:
	'''
	Can quantitatively compare observations
	'''
	def compare(self, observation: 'OBSERVATION', reference: 'OBSERVATION') -> 'VALUE':
		'''
		Compares observations
		:param observation: input observation
		:param reference: input observation (reference)
		:return: quantitative comparison
		'''
		raise NotImplementedError


class Metric(Criterion):
	'''
	Quantiatively measures the distance between observations
	'''
	def measure(self, observation: 'OBSERVATION', reference: 'OBSERVATION') -> 'VALUE':
		'''
		Measures the distance of the observation with respect to the reference
		Distances should adhere to expected properties (e.g. triangle inequality)
		:param observation: input
		:param reference: frame of reference
		:return:
		'''
		raise NotImplementedError

	
	def difference(self, observation: 'OBSERVATION', reference: 'OBSERVATION') -> 'OBSERVATION':
		'''
		Measures the difference of the observation with respect to the reference for each DOF in the observation.
		observation + difference(observation, reference) == reference
		:param observation: input
		:param reference: frame of reference
		:return: difference on a DOF bases
		'''
		raise NotImplementedError
	
	
	def compare(self, observation: 'OBSERVATION', reference: 'OBSERVATION') -> 'VALUE':
		return self.measure(observation, reference)


class Discriminator:
	'''
	Can quantitatively judge the quality of an observation
	'''
	def judge(self, observation: 'OBSERVATION') -> 'VALUE':
		'''
		Evaluates the quality of the observation (a higher value should correspond to a higher quality observation)
		:param observation: input
		:return: quality of the observation
		'''
		raise NotImplementedError


class Integrator:
	'''
	Can integrate a path of observations
	'''
	def integrate(self, start: 'OBSERVATION', end: 'OBSERVATION' = None, steps: int = None) -> 'PATH':
		'''
		
		:param start: starting conditions
		:param end: end conditions (aka target)
		:param steps: number of steps to
		:return: path that was taken by the integration (may be a value if the integral is already evaluated)
		'''
		raise NotImplementedError


class Distortion:
	'''
	Can distort observations
	'''
	def distort(self, observation: 'OBSERVATION') -> 'OBSERVATION':
		'''
		Distorts the given observations
		:param observation: input to be distorted
		:return: distorted observations
		'''


class Compressor(Encoder):
	'''
	Can compress an observation into a bytestring
	'''
	def compress(self, observation: 'OBSERVATION') -> bytes:
		'''
		Compress a given observation (aka message)
		:param observation: input
		:return: bytestring containing information to recover the observation
		'''
		raise NotImplementedError


	def encode(self, observation: 'OBSERVATION') -> 'LATENT':
		return self.compress(observation)


class Decompressor(Decoder):
	'''
	Can recover an observation from a bytestring
	'''
	def decompress(self, data: bytes) -> 'OBSERVATION':
		'''
		Decompress the given data to recover the observation
		:param data: input bytestring
		:return: decompressed observation
		'''
		raise NotImplementedError
	
	
	def decode(self, latent: 'LATENT') -> 'OBSERVATION':
		return self.decompress(latent)