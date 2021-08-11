import torch
import numpy as numpy

from rgn2_replica.utils import *

def test_chunk_permute(): 
	seq = torch.tensor([1,2,3,4,5,6,7,8,9]*3)
	res = chunk_permute(seq)

	assert True

def test_masked_lang(): 
	seq = torch.tensor([1,2,3,4,5,6,7,8,9]*3)
	res = masked_lang(seq)

	assert True


def test_mask_seq(): 
	seq = torch.tensor([1,2,3,4,5,6,7,8,9]*3)
	res = mask_seq(seq)

	assert True