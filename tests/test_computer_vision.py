import math
import random

import numpy
import pytest
from computer_vision import perspective_matrix_from_known_points, refine_camera, decompose_homography, \
	homography_from_planar_projection_basic, decompose_projection_matrix, decompose_homography_svd, \
	homography_from_planar_projection_robust, decompose_unnormalized_homography
from camera import CameraIntrinsics, CameraExtrinsics
from rotation import RotationMatrix


def rot_distance(a, b):
	"""Return the 'difference' between a and b, assuming both are angles."""
	return abs(math.cos(a) - math.cos(b))


def test_homography_identity():
	points = numpy.asarray([
		[0, 0],
		[10, 0],
		[0, 10],
		[10, 10],
		[2, 3],
		[3, 2],
		[4, 4],
		[11, 1],
	])
	cam = CameraExtrinsics()
	projection = cam.project_points(numpy.hstack([points, numpy.ones(shape=(points.shape[0], 1))]))
	h = homography_from_planar_projection_basic(points, projection)
	assert numpy.allclose(h, numpy.eye(3))


def test_homography_randomized():
	for _ in range(100):
		# Build points...
		coplanar_points_on_z = numpy.random.uniform(low=-1, high=1, size=(16, 3))
		coplanar_points_on_z[:, 2] = 1.0
		# Build rotation part of homography.
		rand_homography = numpy.random.uniform(low=-10, high=10, size=(3,3))
		rand_homography /= rand_homography[-1, -1]
		# Project and normalize points.
		projected = (rand_homography @ coplanar_points_on_z.T).T
		projected[:,0] /= projected[:,2]
		projected[:,1] /= projected[:,2]
		projected[:,2] /= projected[:,2]
		# Compute
		basic_homography = homography_from_planar_projection_basic(coplanar_points_on_z, projected)
		robust_homography = homography_from_planar_projection_robust(coplanar_points_on_z, projected)
		basic_homography /= basic_homography[2, 2]
		robust_homography /= robust_homography[2, 2]
		# Evaluate
		assert basic_homography == pytest.approx(rand_homography)
		assert robust_homography == pytest.approx(rand_homography)
		#assert numpy.allclose(our_homography, cv_homography)


def test_homography_translation():
	# Move camera up, back, and left by ten units.
	fake_homography = numpy.eye(3)
	fake_homography[0, 2] = -3
	fake_homography[1, 2] = -5
	points = numpy.random.uniform(low=-1, high=1, size=(16,3))
	points[:, 0] /= points[:, 2]
	points[:, 1] /= points[:, 2]
	points[:, 2] /= points[:, 2]
	transformed = (fake_homography @ points.T).T
	transformed[:, 0] /= transformed[:, 2]
	transformed[:, 1] /= transformed[:, 2]
	transformed[:, 2] /= transformed[:, 2]
	calculated_homography_basic = homography_from_planar_projection_basic(points, transformed)
	calculated_homography_robust = homography_from_planar_projection_robust(points, transformed)
	rotation, translation = decompose_unnormalized_homography(calculated_homography_robust)
	print("Target:")
	print(fake_homography)
	print("Calculated homography:")
	print(calculated_homography_robust)
	print("Recovered rotation:")
	print(rotation)
	print("Recovered translation:")
	print(translation)
	assert translation[0, 0] == pytest.approx(fake_homography[0, 2])
	assert translation[1, 0] == pytest.approx(fake_homography[1, 2])
	assert rotation == pytest.approx(numpy.eye(3))


def test_homography_rotation():
	# Move camera up, back, and left by ten units.
	fake_homography = RotationMatrix.x_rotation(-0.25 * math.pi) @ RotationMatrix.y_rotation(0.1 * math.pi)
	fake_homography[:,2] = 0
	fake_homography[2,2] = 1
	points = numpy.random.uniform(low=-1, high=1, size=(16,3))
	points[:, 0] /= points[:, 2]
	points[:, 1] /= points[:, 2]
	points[:, 2] /= points[:, 2]
	transformed = (fake_homography @ points.T).T
	transformed[:, 0] /= transformed[:, 2]
	transformed[:, 1] /= transformed[:, 2]
	transformed[:, 2] /= transformed[:, 2]
	calculated_homography_basic = homography_from_planar_projection_basic(points, transformed)
	calculated_homography_robust = homography_from_planar_projection_robust(points, transformed)
	calculated_homography_robust /= calculated_homography_robust[2, 2]
	print(fake_homography)
	print(calculated_homography_basic)
	print(calculated_homography_robust)
	assert fake_homography == pytest.approx(calculated_homography_basic)
	assert fake_homography == pytest.approx(calculated_homography_robust)


def test_refine_pose():
	coplanar_points_on_z = numpy.random.uniform(low=-1, high=1, size=(16, 3))
	coplanar_points_on_z[:, 2] = 0.0

	target_intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 1, 1)
	target_extrinsics = CameraExtrinsics(0.1, 0.2, 0.3, 4, 5, 6)
	projection = target_extrinsics.project_points(coplanar_points_on_z, target_intrinsics, renormalize=True)
	estimated_intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 1, 1)
	estimated_extrinsics = CameraExtrinsics(0.1, -0.1, 0.1, -0.1, 0.1, 2.0)
	estimated_intrinsics, estimated_extrinsics = refine_camera(projection, coplanar_points_on_z, estimated_intrinsics, estimated_extrinsics, refine_k=False, refine_rt=True, max_iterations=2000)
	#assert numpy.allclose(target_intrinsics.to_matrix(), estimated_intrinsics.to_matrix())
	assert numpy.allclose(target_extrinsics.to_matrix(), estimated_extrinsics.to_matrix(), rtol=0.4, atol=0.5)


@pytest.mark.parametrize('translation', [(0, 0, -2), (0, 0, 2), (1, 0, -2), (1, 0, 2), (0, 1, -2), (0, 1, 2)])
@pytest.mark.parametrize('rotation', [(0, 0, 0), (0.1, 0.0, 0.0), (0, 0.1, 0), (0, 0, 0.1), (0.1, 0, 0.1)])
def test_compute_perspective(translation, rotation):
	# Build our sample points and intrinsics.
	coplanar_points_on_z = numpy.random.uniform(low=-1, high=1, size=(8, 3))
	# Coerce to planar.
	#coplanar_points_on_z[:,2] = 0.0

	rx, ry, rz = rotation
	tx, ty, tz = translation

	known_extrinsics = CameraExtrinsics(rx, ry, rz, tx, ty, tz)  # We do have to move off the plane for sanity.
	projection = known_extrinsics.project_points(coplanar_points_on_z, renormalize=True)
	assert projection.shape[0] == coplanar_points_on_z.shape[0]
	assert numpy.allclose(perspective_matrix_from_known_points(coplanar_points_on_z, projection), known_extrinsics.to_matrix())


# Can stack as @pytest.mark.parametrize('translation,rotation', [((dx,dy,dz), (rx,ry,rz)), ((a,b,c),(d,e,f)), ])
# Or for all permutations:
@pytest.mark.parametrize('translation', [(0, 0, -2), (0, 0, 2), (1, 0, -2), (1, 0, 2), (0, 1, -2), (0, 1, 2)])
@pytest.mark.parametrize('rotation', [(0, 0, 0), (0.1, 0.0, 0.0), (0, 0.1, 0), (0, 0, 0.1), (0.1, 0, 0.1)])
def test_decompose_perspective(translation, rotation):
	# Build our sample points and intrinsics.
	coplanar_points_on_z = numpy.random.uniform(low=-1, high=1, size=(8, 3))
	# Coerce to planar.
	#coplanar_points_on_z[:,2] = 0.0

	rx, ry, rz = rotation
	tx, ty, tz = translation

	known_extrinsics = CameraExtrinsics(rx, ry, rz, tx, ty, tz)  # We do have to move off the plane for sanity.
	projection = known_extrinsics.project_points(coplanar_points_on_z, renormalize=True)
	assert projection.shape[0] == coplanar_points_on_z.shape[0]
	estimated_intrinsics, estimated_rotation, estimated_translation = decompose_projection_matrix(
		perspective_matrix_from_known_points(coplanar_points_on_z, projection))
	#estimated_intrinsics, estimated_extrinsics = refine_camera(projection, coplanar_points_on_z, estimated_intrinsics, estimated_extrinsics)
	#assert known_extrinsics.to_matrix() == pytest.approx(estimated_extrinsics.to_matrix())
	rot = RotationMatrix.from_matrix(estimated_rotation)
	assert known_extrinsics.rx == pytest.approx(rot.x)
	assert known_extrinsics.ry == pytest.approx(rot.y)
	assert known_extrinsics.rz == pytest.approx(rot.z)
	assert known_extrinsics.x_translation == pytest.approx(estimated_translation[0])
	assert known_extrinsics.y_translation == pytest.approx(estimated_translation[1])
	assert known_extrinsics.z_translation == pytest.approx(estimated_translation[2])
