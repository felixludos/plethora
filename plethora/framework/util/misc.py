




def combine_dims(tensor, start=1, end=None):
	if end is None:
		end = len(tensor.shape)
	combined_shape = [*tensor.shape[:start], -1, *tensor.shape[end:]]
	return tensor.view(*combined_shape)


def split_dim(tensor, *splits, dim=0):
	split_shape = [*tensor.shape[:dim], *splits, *tensor.shape[dim+1:]]
	return tensor.view(*split_shape)


def swap_dim(tensor, d1=0, d2=1):
	dims = list(range(len(tensor.shape)))
	dims[d1], dims[d2] = dims[d2], dims[d1]
	return tensor.permute(*dims)






