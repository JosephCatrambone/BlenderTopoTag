
import math
import numpy
import pytest
from main import TopoTag, convert_image, binarize, find_tags


def test_topotag_decode_k3():
	for i in range(0, (2**7)-1):
		img = TopoTag.generate_marker(3, i, 100)
		tags, island_data, island_matrix = find_tags(binarize(convert_image(img)))
		assert len(tags) == 1
		assert tags[0].tag_id == i
