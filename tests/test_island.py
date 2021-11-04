import cProfile
import os.path

import numpy

from island import flood_fill_connected


def run_connected_component_performance_bench():
	for i in range(100, 1000, 100):
		mat = numpy.random.uniform(size=(i,i)) > 0.0
		mat = mat.astype(numpy.uint8)
		_ = flood_fill_connected(mat)


def test_connected_component_runtime():
	cProfile.runctx('run_connected_component_performance_bench()', globals(), locals(), filename=None)