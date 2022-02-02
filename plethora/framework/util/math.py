# import random
# import numpy as np
import torch


def set_seed(seed=None):
	if seed is None:
		seed = gen_random_seed()
	# random.seed(seed)
	# np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
	return seed


def gen_random_seed(gen=None):
	return torch.randint(-2**63, 2**63-1, size=(), generator=gen).item()


def gen_deterministic_seed(seed):
	return gen_random_seed(torch.Generator().manual_seed(seed))