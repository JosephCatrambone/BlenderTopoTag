import math
from dataclasses import dataclass

import numpy


@dataclass
class RotationMatrix:
	@classmethod
	def to_euler(cls, r):
		"""Return x, y, z such that this (R) = ZYX"""
		x = math.atan2(r[2,1], r[2,2])
		y = math.atan2(-r[2,0], math.sqrt(r[2,1]*r[2,1] + r[2,2]*r[2,2]))
		z = math.atan2(r[1,0], r[0,0])
		return (x, y, z)

	@classmethod
	def to_matrix(cls, x: float, y: float, z:float):
		#x_rot = cls.from_x_rotation(x)
		#y_rot = cls.from_y_rotation(y)
		#z_rot = cls.from_z_rotation(z)
		#return z_rot @ y_rot @ x_rot
		# To invert, x, y, z = -z, -y, -x
		cx = math.cos(x)
		sx = math.sin(x)
		cy = math.cos(y)
		sy = math.sin(y)
		cz = math.cos(z)
		sz = math.sin(z)
		return numpy.asarray([
			[cz*cy*cx-sz*sx, cz*cy*sx+sz*cx, -cz*sy],
			[-sz*cy*cx-cz*sx, -sz*cy*sx+cz*cx, sz*sy],
			[sy*cx, sy*sx, cy]
		])

	@classmethod
	def from_x_rotation(cls, val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[1, 0, 0],
			[0, c, -s],
			[0, s, c]
		])

	@classmethod
	def from_y_rotation(cls, val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[c, 0, s],
			[0, 1, 0],
			[-s, 0, c]
		])

	@classmethod
	def from_z_rotation(cls, val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[c, -s, 0],
			[s, c, 0],
			[0, 0, 1]
		])