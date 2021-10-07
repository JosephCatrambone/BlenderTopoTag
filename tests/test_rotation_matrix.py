
import numpy
import pytest
from main import RotationMatrix


def test_identity():
	ident = numpy.eye(3)
	assert numpy.allclose(RotationMatrix.from_x_rotation(0), ident)
	assert numpy.allclose(RotationMatrix.from_y_rotation(0), ident)
	assert numpy.allclose(RotationMatrix.from_z_rotation(0), ident)

def test_to_euler_ident():
	x_rot, y_rot, z_rot = RotationMatrix.to_euler(numpy.eye(3))
	assert abs(x_rot) < 1e-6
	assert abs(y_rot) < 1e-6
	assert abs(z_rot) < 1e-6

def test_from_euler_ident():
	rot = RotationMatrix.to_matrix(0, 0, 0)
	assert numpy.allclose(rot, numpy.eye(3))

def test_self_inverse():
	rot = RotationMatrix.to_matrix(10, 20, 30)
	# A rotation matrix transposed must be its own inverse.
	assert numpy.allclose(rot @ rot.T, numpy.eye(3))
