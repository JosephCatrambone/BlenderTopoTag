
import math
import numpy
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
