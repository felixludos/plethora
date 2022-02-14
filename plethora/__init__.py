__version__ = "0.1"

import omnifig as fig

from .datasets.toy import SwissRollDataset


@fig.Script('test')
def _test_script(A):

	dataset = SwissRollDataset().load()

	print(len(dataset))

	print(dataset.get_observation().shape)
	print(dataset.get_observation_space().sample(10).shape)

	print(dataset.get_label().shape)

	pass

