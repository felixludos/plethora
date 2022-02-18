
import torch
from omnibelt import get_printer
from ...datasets.base import SyntheticDataset
from ..base import Task, BatchedTask, ResultsContainer

prt = get_printer(__file__)



class InterpolationTask(BatchedTask):
	def __init__(self, encoder=None, interpolator=None, decoder=None, num_steps=12,
	             score_key=None, **kwargs):
		super().__init__(score_key=score_key, **kwargs)
		
		self.encoder = encoder
		self.interpolator = interpolator
		self.decoder = decoder
		
		self.num_steps = num_steps
		
	
	







