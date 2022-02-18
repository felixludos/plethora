__version__ = "0.1"

import omnifig as fig

from . import tasks
from .datasets import toy


@fig.Script('test')
def _test_script(A):

	dataset = toy.SwissRollDataset(100).load()

	print(len(dataset))

	print(dataset.get_observation().shape)
	print(dataset.get_observation_space().sample(10).shape)

	print(dataset.get_label().shape)

	batches = [batch for batch in dataset]
	for batch in dataset:
		batches.append(batch)
	print([[x.shape for x in batch] for batch in batches])

	out = dataset.subset(0.5)
	print(out.get_observation().shape)

	out = dataset.split([None, 0.2])
	print(out)
	print(out.get_observation().shape)

	pass

