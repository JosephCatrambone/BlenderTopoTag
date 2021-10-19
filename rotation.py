import math
from dataclasses import dataclass

import numpy


@dataclass
class RotationMatrix:
	# To invert, x, y, z = -z, -y, -x

	def __init__(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z

	@classmethod
	def from_zyx_matrix(cls, r):
		"""Return x, y, z such that this (R) = ZYX"""
		x = math.atan2(r[2,1], r[2,2])
		y = math.atan2(-r[2,0], math.sqrt(r[2,1]*r[2,1] + r[2,2]*r[2,2]))
		z = math.atan2(r[1,0], r[0,0])
		return cls(x, y, z)

	def to_matrix(self, order='xyz'):
		rot = numpy.eye(3)
		for c in order.lower():
			if c == 'x':
				rot = rot @ RotationMatrix.x_rotation(self.x)
			elif c == 'y':
				rot = rot @ RotationMatrix.y_rotation(self.y)
			elif c == 'z':
				rot = rot @ RotationMatrix.z_rotation(self.z)
			else:
				raise Exception("Unknown order parameter: %s", c)
		return rot

	@staticmethod
	def to_zyx_matrix(z: float, y: float, x: float):
		"""zyx is read right-to-left.  It means x rotation first, then y, then z.  Assumes p' = zyx p"""
		return RotationMatrix.z_rotation(z) @ RotationMatrix.y_rotation(y) @ RotationMatrix.x_rotation(x)

	@staticmethod
	def x_rotation(val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[1, 0, 0],
			[0, c, -s],
			[0, s, c]
		])

	@staticmethod
	def y_rotation(val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[c, 0, s],
			[0, 1, 0],
			[-s, 0, c]
		])

	@staticmethod
	def z_rotation(val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[c, -s, 0],
			[s, c, 0],
			[0, 0, 1]
		])