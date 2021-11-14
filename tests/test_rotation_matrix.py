
import math
import numpy
import pytest
import random
from rotation import RotationMatrix


def test_identity():
	ident = numpy.eye(3)
	assert numpy.allclose(RotationMatrix.x_rotation(0), ident)
	assert numpy.allclose(RotationMatrix.y_rotation(0), ident)
	assert numpy.allclose(RotationMatrix.z_rotation(0), ident)

def test_to_euler_ident():
	rot = RotationMatrix.from_matrix(numpy.eye(3))
	assert abs(rot.x) < 1e-6
	assert abs(rot.y) < 1e-6
	assert abs(rot.z) < 1e-6

def test_sample_output_a():
	rot = RotationMatrix(-0.1, 0.2, 0.3).to_matrix()
	expected = numpy.asarray([
		[0.93, -0.31, 0.2],
		[0.29, 0.95, 0.1],
		[-0.22, -0.03, 0.98]
	])
	numpy.allclose(rot, expected)

def test_sample_output_b():
	rot = RotationMatrix(3.14, -1, 2).to_matrix()
	expected = numpy.asarray([
		[-0.23, -0.49, 0.84],
		[-0.91, 0.42, 0.0],
		[-0.35, -0.77, -0.54]
	])
	numpy.allclose(rot, expected)

def test_from_euler_ident():
	rot = RotationMatrix(0, 0, 0).to_matrix()
	assert numpy.allclose(rot, numpy.eye(3))

def test_self_inverse():
	rot = RotationMatrix(10, 20, 30).to_matrix()
	# A rotation matrix transposed must be its own inverse.
	assert numpy.allclose(rot @ rot.T, numpy.eye(3))
