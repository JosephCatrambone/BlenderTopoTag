import numpy
from typing import Tuple

from camera import CameraIntrinsics, CameraExtrinsics
from image_processing import Matrix
from rotation import RotationMatrix


def calibrate_camera_from_known_points(projection: Matrix, world: Matrix) -> (CameraIntrinsics, CameraExtrinsics):
	"""Compute the projection matrix from the projected image of the world coordinates."""
	assert projection.shape[1] >= 2
	assert world.shape[1] >= 3
	# s * [u', v', 1].T = [R | t] * [x, y, z, 1].T
	#
	#   0  1  2  3  4  5  6  7    8     9     10   11
	# | x  y  z  1  0  0  0  0  -u'x  -u'y  -u'z  -u' |  * [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz].T = 0
	# | 0  0  0  0  x  y  z  1  -v'x  -v'y  -v'z  -v' |
	# Use SVD to find r|t.
	#
	# Full derivation:
	# [x, y, z].T = [p1, p2, p3, p4; p5, p6, p7, p8; p9, p10, p11, p12] * [X, Y, Z, 1].T
	# xProj = [P1 (p1, p2, p3, p4); P2; P3] * X.T
	# x' = P1 X / P3 X
	# y' = P2 X / P3 X
	# -> Make linear via manip.
	# P2 X - P3 X y' = 0
	# P1 X - P3 X x' = 0
	# ->
	# [ X.T, 0, -x'*X.T ] * [ P1.T ] = 0
	# [ 0, X.T, -y'*X.T ]   [ P2.T ]
	#                       [ P3.T ]
	# A                     x
	homo_mat = numpy.zeros(shape=(projection.shape[0]*2, 12))
	for i in range(projection.shape[0]):
		x = world[i, 0]
		y = world[i, 1]
		z = world[i, 2]
		u = projection[i, 0]
		v = projection[i, 1]
		homo_mat[(i * 2), 0] = x
		homo_mat[(i * 2), 1] = y
		homo_mat[(i * 2), 2] = z
		homo_mat[(i * 2), 3] = 1

		homo_mat[(i * 2), 8] = -u*x
		homo_mat[(i * 2), 9] = -u*y
		homo_mat[(i * 2), 10] = -u*z
		homo_mat[(i * 2), 11] = -u

		homo_mat[(i * 2)+1, 4] = x
		homo_mat[(i * 2)+1, 5] = y
		homo_mat[(i * 2)+1, 6] = z
		homo_mat[(i * 2)+1, 7] = 1

		homo_mat[(i * 2)+1, 8] = -v*x
		homo_mat[(i * 2)+1, 9] = -v*y
		homo_mat[(i * 2)+1, 10] = -v*z
		homo_mat[(i * 2)+1, 11] = -v
	# Given Ax=0, A is an overdetermined homogeneous solution, and the nontrivial solution is the smallest eigenvec.
	_, _, v = numpy.linalg.svd(homo_mat.T @ homo_mat)
	#_, _, v = numpy.linalg.svd(homo_mat, full_matrices=False)
	p = v[-1,:].reshape((3,4))

	# P = [p1, p2, p3, p4; p5, p6, p7, p8; p9, p10, p11, p12]
	# R = [p1, p2, p3; p5, p6, p7; p9, p10, p11]
	# t = [p4; p8; p12]
	# P = K[R|t]
	# P = K[R| -Rc]
	# P = M| -Mc
	# M = KR -> K is right-upper-triangular, R is orthogonal.  Get via RQ decomposigion.

	_, _, v = numpy.linalg.svd(p, full_matrices=False)
	C = v[:,-1]
	if abs(v[-1,-1]) > 1e-6:
		C /= v[-1, -1]

	M = p[0:3,0:3]

	# Renormalize p by multiplying by sign(det(M)).
	sign_det_m = 1
	if numpy.linalg.det(M) < 0:
		sign_det_m = -1
	M *= sign_det_m
	p *= sign_det_m

	K_hat, R_hat = numpy.linalg.qr(M)
	K_signs = numpy.sign(K_hat)
	D = numpy.diag([K_signs[0,0], K_signs[1,1], K_signs[2,2]])
	K = K_hat @ D
	R = D @ R_hat
	t = -R @ C

	K /= K[2,2]

	intrinsics = CameraIntrinsics(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2])
	x_rot, y_rot, z_rot = RotationMatrix.to_euler(R)
	extrinsics = CameraExtrinsics(x_rot, y_rot, z_rot, t[0], t[1], t[2])
	return intrinsics, extrinsics


def fundamental_from_correspondences(p: Matrix, q: Matrix) -> Matrix:
	"""Given two sets of points in 3D space, use the normalized eight points algorithm to return the fundamental matrix."""
	assert p.shape[1] == q.shape[1]
	a_mat = numpy.zeros((p.shape[1], 9))
	for i in range(p.shape[1]):
		# x'x, x'y, x', y'x, y'y, y', x, y, 1
		px = p[i,0]
		py = p[i,1]
		qx = q[i,0]
		qy = q[i,1]
		a_mat[i,0] = px*qx
		a_mat[i,1] = px*qy
		a_mat[i,2] = px
		a_mat[i,3] = py*qx
		a_mat[i,4] = py*qy
		a_mat[i,5] = py
		a_mat[i,6] = qx
		a_mat[i,7] = qy
		a_mat[i,8] = 1.0
	u, s, v = numpy.linalg.svd(a_mat)
	fundamental = v[-1].reshape((3,3))

	u, s, v = numpy.linalg.svd(fundamental)
	s[2] = 0
	fundamental = u @ (numpy.diag(s) @ v)
	return fundamental


def homography_from_planar_projection_basic(projection: Matrix, world_plane: Matrix) -> Matrix:
	"""Compute the 3x3 homography matrix with h33 == 1.0, assuming world_plane is on the plane xy-plane with z=0."""
	assert projection.shape[0] == world_plane.shape[0]

	# 2nx8 * 8x1 = 2nx1
	a_mat = numpy.zeros(shape=(2*projection.shape[0], 8))
	b_mat = numpy.zeros(shape=(2*projection.shape[0], 1))
	for idx in range(projection.shape[0]):
		x_w = world_plane[idx, 0]
		y_w = world_plane[idx, 1]
		x_p = projection[idx, 0]
		y_p = projection[idx, 1]
		#  0  1  2  3  4  5    6      7
		# [x, y, 1, 0, 0, 0, -x*x', -y*x'] * g = [x']
		# [0, 0, 0, x, y, 1, -x*y', -y*y']     = [y']
		a_mat[(idx * 2) + 0, 0] = x_w
		a_mat[(idx * 2) + 0, 1] = y_w
		a_mat[(idx * 2) + 0, 2] = 1
		a_mat[(idx * 2) + 1, 3] = x_w
		a_mat[(idx * 2) + 1, 4] = y_w
		a_mat[(idx * 2) + 1, 5] = 1

		a_mat[(idx * 2) + 0, 6] = -x_w*x_p
		a_mat[(idx * 2) + 0, 7] = -y_w*x_p
		a_mat[(idx * 2) + 1, 6] = -x_w*y_p
		a_mat[(idx * 2) + 1, 7] = -y_w*y_p

		b_mat[(idx * 2) + 0, 0] = x_p
		b_mat[(idx * 2) + 1, 0] = y_p
	soln, residuals, rank, singular_values = numpy.linalg.lstsq(a_mat, b_mat)
	homography = numpy.asarray([
		[soln[0,0], soln[1,0], soln[2,0]],
		[soln[3,0], soln[4,0], soln[5,0]],
		[soln[6,0], soln[7,0], 1],
	])
	return homography


def homography_from_planar_projection_robust(projection: Matrix, world_plane: Matrix) -> Matrix:
	"""Compute the 3x3 homography matrix, assuming world_plane is on the plane xy-plane with z=0.  Slower than basic,
	but still works if the axis is inside the frame or h33 is at infinity."""
	assert projection.shape[0] == world_plane.shape[0]
	# 2nx9 * 9x1 = zeros
	a_mat = numpy.zeros(shape=(2*projection.shape[0], 9))
	for idx in range(projection.shape[0]):
		x_w = world_plane[idx, 0]
		y_w = world_plane[idx, 1]
		x_p = projection[idx, 0]
		y_p = projection[idx, 1]
		#  0  1  2  3  4  5    6      7
		# [x, y, 1, 0, 0, 0, -x*x', -y*x'] * g = [x']
		# [0, 0, 0, x, y, 1, -x*y', -y*y']     = [y']
		a_mat[(idx * 2) + 0, 0] = x_w
		a_mat[(idx * 2) + 0, 1] = y_w
		a_mat[(idx * 2) + 0, 2] = 1
		a_mat[(idx * 2) + 1, 3] = x_w
		a_mat[(idx * 2) + 1, 4] = y_w
		a_mat[(idx * 2) + 1, 5] = 1

		a_mat[(idx * 2) + 0, 6] = -x_w*x_p
		a_mat[(idx * 2) + 0, 7] = -y_w*x_p
		a_mat[(idx * 2) + 0, 8] = -x_p
		a_mat[(idx * 2) + 1, 6] = -x_w*y_p
		a_mat[(idx * 2) + 1, 7] = -y_w*y_p
		a_mat[(idx * 2) + 1, 8] = -y_p

	a_mat = a_mat.T @ a_mat
	assert a_mat.shape[0] == 9 and a_mat.shape[1] == 9

	u, s, v = numpy.linalg.svd(a_mat.T @ a_mat)
	#_, _, h = numpy.linalg.svd(homo_mat, full_matrices=False)
	homography = v[-1, :].reshape((3, 3))
	return homography


def refine_camera(projected_points: Matrix, world_points: Matrix, intrinsic: CameraIntrinsics, extrinsic: CameraExtrinsics, max_iterations: 1000, epsilon: 1e-6) -> Tuple[CameraIntrinsics, CameraExtrinsics]:
	# Project the points forward into 3d space at an assumed depth and compute the error.
	# p = K R|t w
	reprojection = extrinsic.project_points(world_points, intrinsic, renormalize=True)
	error = projected_points[:, 0:2] - reprojection[:, 0:2]
	adjoint = numpy.ones_like(error)
