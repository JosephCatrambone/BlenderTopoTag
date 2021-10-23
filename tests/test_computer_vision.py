import math
import numpy
import pytest
from computer_vision import perspective_matrix_from_known_points, refine_camera, decompose_homography, homography_from_planar_projection_basic, decompose_projection_matrix
from camera import CameraIntrinsics, CameraExtrinsics

def rot_distance(a, b):
	"""Return the 'difference' between a and b, assuming both are angles."""
	return abs(math.cos(a) - math.cos(b))

def test_homography_round_trip():
	intrinsics = CameraIntrinsics(1, 1, 0, 0, 0)
	points = numpy.asarray([
		[0, 0],
		[10, 0],
		[0, 10],
		[10, 10],
		[2, 1],
		[1, 2],
		[5, 5],
		[7, 9],
		[9, 7],
	])
	camera_positions = [
		(0, 0, 0),
		(0, 0, -10),
		(0, 0, 10),
		(0, 3, 0),
		(1, 0, 0),
		(1, 5, 0),
		(10, 5, 2),
	]
	camera_rotations = [
		(0, 0, 0),
		(-0.1, 0, 0),
		(0.1, 0, 0),
		(0, -0.1, 0),
		(0, 0.1, 0),
		(0, 0, -0.1),
		(0, 0, 0.1),
	]
	for rx, ry, rz in camera_rotations:
		for tx, ty, tz in camera_positions:
			cam = CameraExtrinsics(rx, ry, rz, tx, ty, tz)
			projection = cam.project_points(numpy.hstack([points, numpy.ones(shape=(points.shape[0], 1))]))
			est_cam = decompose_homography(homography_from_planar_projection_basic(projection, points), intrinsics)
			est_projection = est_cam.project_points(numpy.hstack([points, numpy.ones(shape=(points.shape[0], 1))]))
			diff = projection - est_projection
			#assert numpy.allclose(diff[:,0:2], numpy.zeros_like(projection[:,0:2]), rtol=1e-4, atol=1e-4)
			assert rot_distance(est_cam.y_rotation, ry) < 1e-2
			assert rot_distance(est_cam.z_rotation, rz) < 1e-2
			assert abs(est_cam.x_translation - tx) < 1e-4
			assert abs(est_cam.y_translation - ty) < 1e-4
			#assert numpy.allclose(est_cam.z_translation, tz)
			#assert numpy.allclose(est_cam.to_matrix(), cam.to_matrix())

def test_refine_pose():
	coplanar_points_on_z = numpy.random.uniform(low=-1, high=1, size=(16, 3))
	coplanar_points_on_z[:, 2] = 0.0

	target_intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 1, 1)
	target_extrinsics = CameraExtrinsics(0.1, 0.2, 0.3, 4, 5, 6)
	projection = target_extrinsics.project_points(coplanar_points_on_z, target_intrinsics, renormalize=True)
	estimated_intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 1, 1)
	estimated_extrinsics = CameraExtrinsics(0.1, -0.1, 0.1, -0.1, 0.1, 2.0)
	estimated_intrinsics, estimated_extrinsics = refine_camera(projection, coplanar_points_on_z, estimated_intrinsics, estimated_extrinsics, refine_k=False, refine_rt=True)
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
	estimated_intrinsics, estimated_extrinsics = decompose_projection_matrix(
		perspective_matrix_from_known_points(coplanar_points_on_z, projection))
	#estimated_intrinsics, estimated_extrinsics = refine_camera(projection, coplanar_points_on_z, estimated_intrinsics, estimated_extrinsics)
	#assert known_extrinsics.to_matrix() == pytest.approx(estimated_extrinsics.to_matrix())
	assert known_extrinsics.x_rotation == pytest.approx(estimated_extrinsics.x_rotation)
	assert known_extrinsics.y_rotation == pytest.approx(estimated_extrinsics.y_rotation)
	assert known_extrinsics.z_rotation == pytest.approx(estimated_extrinsics.z_rotation)
	assert known_extrinsics.x_translation == pytest.approx(estimated_extrinsics.x_translation)
	assert known_extrinsics.y_translation == pytest.approx(estimated_extrinsics.y_translation)
	assert known_extrinsics.z_translation == pytest.approx(estimated_extrinsics.z_translation)
