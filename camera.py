import math
from dataclasses import dataclass
from typing import NewType, Optional

import numpy

from image_processing import Matrix
from rotation import RotationMatrix

@dataclass
class CameraIntrinsics:
	focal_length_x: float
	focal_length_y: float
	skew: float
	principal_point_x: int
	principal_point_y: int

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
	x_rotation: float
	y_rotation: float
	z_rotation: float
	x_translation: float
	y_translation: float
	z_translation: float  # The camera looks forward to -Z when all rotation is zero.

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
		points_2d = (projection @ points_3d.T).T

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
		rotation_matrix = RotationMatrix.to_matrix(self.x_rotation, self.y_rotation, self.z_rotation)
		# When multiplying by the 3x4 we rotate, then translate, so when we invert we un-translate, then rotate.
		points[0, :] -= self.x_translation
		points[1, :] -= self.y_translation
		points[2, :] -= self.z_translation
		return (rotation_matrix.T @ points).T

	def to_matrix(self):
		return numpy.hstack([
			RotationMatrix.to_matrix(self.x_rotation, self.y_rotation, self.z_rotation),
			numpy.asarray([[self.x_translation, self.y_translation, self.z_translation]]).T
		])

	def to_inverse_matrix(self):
		"""Return the inverse of this operation."""
		return numpy.hstack([
			RotationMatrix.to_matrix(self.x_rotation, self.y_rotation, self.z_rotation).T,
			-numpy.asarray([[self.x_translation, self.y_translation, self.z_translation]]).T
		])

	@classmethod
	def from_projection_matrix(cls, projection: Matrix):
		assert projection.shape[0] == 3 and projection.shape[1] == 4
		translation = projection[:,-1]
		rotation = projection[0:3,0:3]
		x_rot, y_rot, z_rot = RotationMatrix.to_euler(rotation)
		return cls(x_rot, y_rot, z_rot, translation[0], translation[1], translation[2])