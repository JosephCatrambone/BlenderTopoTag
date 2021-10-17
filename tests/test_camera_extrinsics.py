
import math
import numpy
import pytest
from computer_vision import calibrate_camera_from_known_points, refine_camera, decompose_homography, homography_from_planar_projection_basic
from camera import CameraIntrinsics, CameraExtrinsics


def rot_distance(a, b):
	"""Return the 'difference' between a and b, assuming both are angles."""
	return abs(math.cos(a) - math.cos(b))

def test_projection_inline():
	cam = CameraExtrinsics(0, 0, 0, 0, 0, 0)
	points_3d = numpy.asarray([
		[0, 0, 0],
		[0, 0, 1],
		[0, 0, 2],
	])
	# We expect points all lined up with the camera to project to 0, 0.
	points_2d = cam.project_points(points_3d)
	assert points_2d.shape[0] == 3
	assert numpy.allclose(points_2d[:,0:2], numpy.asarray([
		[0, 0],
		[0, 0],
		[0, 0],
	]))

def test_projection_rotate_about_z():
	cam = CameraExtrinsics(0, 0, math.pi, 0, 0, 0)
	points_3d = numpy.asarray([
		[0, 0, 10],
		[10, 0, 10],
		[0, 10, 0],
	])
	points_2d = cam.project_points(points_3d)
	assert points_2d.shape[0] == points_3d.shape[0]
	assert numpy.allclose(points_2d[:, 0:2], numpy.asarray([
		[0, 0],
		[-10, 0],
		[0, -10],
	]))

def test_projection():
	cam_extrinsics = CameraExtrinsics(0, 0, 0, 0, 0, 1)
	points_3d = numpy.asarray([
		[0, 0, 0, 1],
		[1, 0, 1, 1],
		[1, 0, 2, 1],
		[1, 0, 3, 1],
		[0, 1, 1, 1],
		[0, 1, 2, 1],
		[0, 1, 3, 1],
	])
	points_2d = cam_extrinsics.project_points(points_3d, renormalize=True)
	assert numpy.allclose(
		points_2d,
		numpy.asarray([
			[0., 0., 1.],
			[0.5, 0., 1.],
			[0.33333333, 0., 1.],
			[0.25, 0., 1.],
			[0., 0.5, 1.],
			[0., 0.33333333, 1.],
			[0., 0.25, 1.]
		])
	)

def test_invert_transform():
	cam_intrinsics = CameraIntrinsics(1, 1, 0, 0, 0)
	cam = CameraExtrinsics(math.pi, math.pi/2, -math.pi/2, 1, 2, 3)
	points_3d = numpy.random.uniform(low=-10, high=10, size=(30, 3))
	points_2d = cam.project_points(points_3d)
	unprojected = cam.unproject_points(points_2d, cam_intrinsics)
	assert numpy.allclose(points_3d, unprojected)

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

def test_compute_extrinsic_calculation():
	# Build our sample points and intrinsics.
	coplanar_points_on_z = numpy.random.uniform(low=-1, high=1, size=(8, 3))
	coplanar_points_on_z[:,2] = 0.0

	# Test for identity.
	known_extrinsics = CameraExtrinsics(0, 0, 0, 0, 0, 10)  # We do have to move off the plane for sanity.
	projection = known_extrinsics.project_points(coplanar_points_on_z, renormalize=True)
	assert projection.shape[0] == coplanar_points_on_z.shape[0]
	estimated_intrinsics, estimated_extrinsics = calibrate_camera_from_known_points(projection, coplanar_points_on_z)
	estimated_intrinsics, estimated_extrinsics = refine_camera(projection, coplanar_points_on_z, estimated_intrinsics, estimated_extrinsics)
	assert numpy.allclose(known_extrinsics.x_rotation, estimated_extrinsics.x_rotation)
	assert numpy.allclose(known_extrinsics.y_rotation, estimated_extrinsics.y_rotation)
	assert numpy.allclose(known_extrinsics.z_rotation, estimated_extrinsics.z_rotation)
	assert numpy.allclose(known_extrinsics.x_translation, estimated_extrinsics.x_translation)
	assert numpy.allclose(known_extrinsics.y_translation, estimated_extrinsics.y_translation)
	#assert numpy.allclose(known_extrinsics.z_translation, estimated_extrinsics.z_translation)

	# Test for pure translation.
	known_extrinsics = CameraExtrinsics(0, 0, 0, 2, 3, 4)
	projection = known_extrinsics.project_points(coplanar_points_on_z, renormalize=True)
	estimated_intrinsics, estimated_extrinsics = calibrate_camera_from_known_points(projection, coplanar_points_on_z)
	assert numpy.allclose(known_extrinsics.x_rotation, estimated_extrinsics.x_rotation)
	assert numpy.allclose(known_extrinsics.y_rotation, estimated_extrinsics.y_rotation)
	assert numpy.allclose(known_extrinsics.z_rotation, estimated_extrinsics.z_rotation)
	assert numpy.allclose(known_extrinsics.x_translation, estimated_extrinsics.x_translation)
	assert numpy.allclose(known_extrinsics.y_translation, estimated_extrinsics.y_translation)
	#assert numpy.allclose(known_extrinsics.to_matrix(), estimated_extrinsics.to_matrix())

	# Test for pure rotation
	known_extrinsics = CameraExtrinsics(0.1, -0.2, 0.3, 0, 0, 10)
	projection = known_extrinsics.project_points(coplanar_points_on_z, renormalize=True)
	estimated_intrinsics, estimated_extrinsics = calibrate_camera_from_known_points(projection, coplanar_points_on_z)
	assert numpy.allclose(known_extrinsics.x_rotation, estimated_extrinsics.x_rotation)
	assert numpy.allclose(known_extrinsics.y_rotation, estimated_extrinsics.y_rotation)
	assert numpy.allclose(known_extrinsics.z_rotation, estimated_extrinsics.z_rotation)
	assert numpy.allclose(known_extrinsics.x_translation, estimated_extrinsics.x_translation)
	assert numpy.allclose(known_extrinsics.y_translation, estimated_extrinsics.y_translation)
	#assert numpy.allclose(known_extrinsics.to_matrix(), estimated_extrinsics.to_matrix())



