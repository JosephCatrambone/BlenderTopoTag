import math
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy

from image_processing import Matrix
from rotation import Quaternion, RotationMatrix

@dataclass
class CameraIntrinsics:
	focal_length_x: float = 1.0
	focal_length_y: float = 1.0
	skew: float = 0.0
	principal_point_x: int = 0
	principal_point_y: int = 0

	def to_matrix(self):
		return numpy.asarray([
			[self.focal_length_x,           self.skew, self.principal_point_x],
			[                  0, self.focal_length_y, self.principal_point_y],
			[                  0,                   0,                      1],
		])

	def to_inverse_matrix(self):
		"""Return the matrix M such that M @ self.to_matrix() = I"""
		# [ACD][acd]   [1  ]
		# [ BE][ be] = [ 1 ]
		# [  1][  f]   [  1]
		return numpy.linalg.inv(self.to_matrix())  # TODO: Compute this with our special knowledge.
		# A*a = 1
		# A*c + C*b = 0
		# Ad + Ce + Df = 0

	@classmethod
	def from_beta_matrix(cls, b):
		principal_point_y = (b[0,1]*b[0,2] - b[0,0]*b[1,2])/(b[0,0]*b[1,1]-b[0,1]*b[0,1])
		scale = b[2,2] - (b[0,2]*b[0,2] + principal_point_y*(b[0,1]*b[0,2]-b[0,0]*b[1,2]))/b[0,0]
		focal_length_x = math.sqrt(scale / b[0,0])
		focal_length_y = math.sqrt(scale * b[0,0] / (b[0,0]*b[1,1] - b[0,1]*b[0,1]))
		skew = -b[0,1]*focal_length_x*focal_length_x*focal_length_y/scale
		principal_point_x = skew*principal_point_y/focal_length_y - b[0,2]*focal_length_x*focal_length_x/skew
		return CameraIntrinsics(focal_length_x, focal_length_y, skew, principal_point_x, principal_point_y)


@dataclass
class CameraExtrinsics:
	rx: float = 0.0
	ry: float = 0.0
	rz: float = 0.0
	tx: float = 0.0
	ty: float = 0.0
	tz: float = 0.0

	def project_points(self, points_3d: Matrix, camera_intrinsics: Optional[CameraIntrinsics] = None, renormalize: bool = False) -> Matrix:
		"""Projects a matrix of size [nx3] OR [nx4] to [nx3].  If renormalize is True, will return [[x, y, 1], ...]."""
		# points is nx3 or nx4.
		if points_3d.shape[1] == 3: # x, y, z.  Augment to x, y, z, 1.
			points_3d = numpy.hstack([points_3d, numpy.ones(shape=(points_3d.shape[0], 1))])
		if points_3d.shape[1] != 4:
			# We attempted to normalize, but obviously we got points that were outside the range.
			raise Exception(f"Got points_3d matrix of unexpected shape: {points_3d.shape}")
		# points = nx4

		# p' = K * H * p
		# K = 3x3, H = 3x4, p = nx4
		projection = self.to_matrix()
		if camera_intrinsics is not None:
			projection = camera_intrinsics.to_matrix() @ projection
		points_2d = (projection @ points_3d.T).T  # Should be a 3xN matrix.

		if renormalize:
			points_2d[:, 0] /= points_2d[:, 2]
			points_2d[:, 1] /= points_2d[:, 2]
			points_2d[:, 2] /= points_2d[:, 2]

		return points_2d

	def unproject_points(self, points_2d: Matrix, camera_intrinsics: CameraIntrinsics) -> Matrix:
		"""Given camera intrinsics and points in the shape [nx2] or [nx3] invert this matrix and find the original 3d points.  Returns an nx3 matrix."""
		# p' = K * H * p
		# K_inv p' = H * p
		# H_inv K_inv p' = p
		# K = 3x3, H = 3x4, p=4xn
		# K_inv = 3x3 -> p' = 3xn
		if points_2d.shape[1] == 2:
			points_2d = numpy.hstack([points_2d, numpy.ones(shape=(points_2d.shape[0], 1))])
		if points_2d.shape[1] != 3:
			raise Exception(f"Got points_2d array with unexpected shape: {points_2d.shape}")
		points = camera_intrinsics.to_inverse_matrix() @ points_2d.T  # Points is now 3xn
		rotation_matrix = RotationMatrix(self.rx, self.ry, self.rz).to_matrix()
		# When multiplying by the 3x4 we rotate, then translate, so when we invert we un-translate, then rotate.
		points[0, :] -= self.tx
		points[1, :] -= self.ty
		points[2, :] -= self.tz
		return (rotation_matrix.T @ points).T

	def to_matrix(self):
		return numpy.hstack([
			RotationMatrix(self.rx, self.ry, self.rz).to_matrix(),
			numpy.asarray([[self.tx], [self.ty], [self.tz]])
		])

	def to_inverse_matrix(self):
		"""Return the inverse of this operation."""
		return numpy.hstack([
			RotationMatrix(self.rx, self.ry, self.rz).to_matrix().T,
			-numpy.asarray([[self.tx], [self.ty], [self.tz]])
		])

	@classmethod
	def from_rotation_and_translation(cls, rotation: Matrix, translation: Matrix):
		translation = translation.reshape(-1)
		rot = RotationMatrix.from_zyx_matrix(rotation)
		return cls(rot.x, rot.y, rot.z, translation[0], translation[1], translation[2])