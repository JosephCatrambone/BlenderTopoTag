
import math
import numpy
import pytest
from computer_vision import calibrate_camera_from_known_points, refine_camera
from camera import CameraIntrinsics, CameraExtrinsics


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
	cam_intrinsics = CameraIntrinsics(1, 1, 0, 0, 0)
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



