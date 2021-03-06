import math
from typing import NewType

import numpy

Matrix = NewType('Matrix', numpy.ndarray)


def make_threshold_map(input_matrix: Matrix) -> Matrix:  # -> grey image matrix
	"""This is basically just blur."""
	# Downscale by four.
	resized = fast_downscale(input_matrix, step=4)
	# Average / blur pixels.
	blurred = blur(resized)
	# Resize twice
	threshold = fast_upscale(fast_upscale(blurred))  # Two downsteps, so two up-steps.
	#threshold = resize_linear(blurred, input_matrix.shape[0], input_matrix.shape[1]) * 0.5
	return threshold


def binarize(image_matrix: Matrix) -> Matrix:
	"""Return a binary integer matrix with ones and zeros."""
	# Should we just combine this with the make_threshold_map function?
	threshold_map = make_threshold_map(image_matrix)
	return (image_matrix >= (threshold_map*0.3 + 0.2)).astype(int)
	#return (image_matrix > image_matrix.mean()+image_matrix.std()).astype(int)


def blur(mat, kernel_width=3):
	center_y = mat.shape[0]//2
	center_x = mat.shape[1]//2
	filter = numpy.zeros_like(mat)
	filter[center_y-kernel_width:center_y+kernel_width, center_x-kernel_width:center_x+kernel_width] = 1.0/(4*kernel_width*kernel_width)
	return fft_convolve2d(mat, filter)


def erode(mat: Matrix, min_nbrs: int = 8) -> Matrix:
	center_y = mat.shape[0]//2
	center_x = mat.shape[1]//2
	filter = numpy.zeros_like(mat)
	filter[center_y-1:center_y+2, center_x-1:center_x+2] = 1.0
	result = fft_convolve2d(mat, filter)
	result = (result > min_nbrs).astype(numpy.uint8)
	return result


def fft_convolve2d(mat, filter):
	"""2D convolution, using FFT.  Convolution has to be at the center of a zeros-like matrix of equal size to the input."""
	fr = numpy.fft.fft2(mat)
	fr2 = numpy.fft.fft2(numpy.flipud(numpy.fliplr(filter)))
	m, n = fr.shape
	cc = numpy.real(numpy.fft.ifft2(fr*fr2))
	cc = numpy.roll(cc, -m//2+1, axis=0)
	cc = numpy.roll(cc, -n//2+1, axis=1)
	return cc


def fast_downscale(image_matrix, step=2):
	return image_matrix[::step, ::step]


def fast_upscale(image_matrix):
	out = numpy.zeros(shape=(image_matrix.shape[0]*2, image_matrix.shape[1]*2))
	out[0::2, 0::2] = image_matrix[:, :]
	interpolated = (image_matrix[:-1,:-1]+image_matrix[1:,1:])/2
	out[0:-2:2, 1:-2:2] = interpolated[:, :]
	out[1:-2:2, 0:-2:2] = interpolated[:, :]
	out[1:-2:2, 1:-2:2] = interpolated[:, :]
	return out


def resize_linear(image_matrix, new_height: int, new_width: int):
	"""Perform a pure-numpy linear-resampled resize of an image."""
	output_image = numpy.zeros((new_height, new_width), dtype=image_matrix.dtype)
	original_height, original_width = image_matrix.shape
	inv_scale_factor_y = original_height/new_height
	inv_scale_factor_x = original_width/new_width

	# This is an ugly serial operation.
	for new_y in range(new_height):
		for new_x in range(new_width):
			# If you had a color image, you could repeat this with all channels here.
			# Find sub-pixels data:
			old_x = new_x * inv_scale_factor_x
			old_y = new_y * inv_scale_factor_y
			x_fraction = old_x - math.floor(old_x)
			y_fraction = old_y - math.floor(old_y)

			# Sample four neighboring pixels:
			left_upper = image_matrix[math.floor(old_y), math.floor(old_x)]
			right_upper = image_matrix[math.floor(old_y), min(image_matrix.shape[1] - 1, math.ceil(old_x))]
			left_lower = image_matrix[min(image_matrix.shape[0] - 1, math.ceil(old_y)), math.floor(old_x)]
			right_lower = image_matrix[min(image_matrix.shape[0] - 1, math.ceil(old_y)), min(image_matrix.shape[1] - 1, math.ceil(old_x))]

			# Interpolate horizontally:
			blend_top = (right_upper * x_fraction) + (left_upper * (1.0 - x_fraction))
			blend_bottom = (right_lower * x_fraction) + (left_lower * (1.0 - x_fraction))
			# Interpolate vertically:
			final_blend = (blend_top * y_fraction) + (blend_bottom * (1.0 - y_fraction))
			output_image[new_y, new_x] = final_blend

	return output_image


def equalize_histogram(mat: Matrix) -> Matrix:
	if mat.mean() < 0.5:
		mat *= 2.0 - mat.mean()
	if mat.mean() < 0.5:
		mat *= 2.0 - mat.mean()
	return mat