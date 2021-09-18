# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# Press the green button in the gutter to run the script.

import math
import numpy
from dataclasses import dataclass

#
# Image processing helpers:
#

def blur(mat, kernel_width=3):
	center_y = mat.shape[0]//2
	center_x = mat.shape[1]//2
	filter = numpy.zeros_like(mat)
	filter[center_y-kernel_width:center_y+kernel_width, center_x-kernel_width:center_x+kernel_width] = 1.0/(4*kernel_width*kernel_width)
	return fft_convolve2d(mat, filter)

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

def resize_linear(image_matrix, new_height:int, new_width:int):
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

#
# Maths + Logic Helpers
#

@dataclass
class IslandBounds:
	id: int = -1
	num_pixels: int = 0
	x_min: int = 0
	y_min: int = 0
	x_max: int = 0
	y_max: int = 0

	def contains(self, other) -> bool:
		"""Returns True if other (an IslandBounds instance) is entirely inside this."""
		if other.x_min <= self.x_min:
			return False
		if other.x_max >= self.x_max:
			return False
		if other.y_min <= self.y_min:
			return False
		if other.y_max >= self.y_max:
			return False
		return True

	def update_from_coordinate(self, x: int, y: int):
		self.num_pixels += 1
		self.x_min = min(self.x_min, x)
		self.y_min = min(self.y_min, y)
		self.x_max = max(self.x_max, x)
		self.y_max = max(self.y_max, y)

def flood_fill_connected(mat, untagged_class: int = 1):
	"""Takes a black and white matrix with 0 as 'empty' and connect components with value==untagged_class (default: 1).
	Returns a tuple with two items:
	 - int matrix with every pixel assigned to a unique class from 2 to n.
	 - A list of length(n+2) where class_n is the position of the bounds information in the list.

	Example:
		matrix[5, 3] == 18  # The pixel at x=3, y=5 is a member of class 18.
		bounds = islands[18]
		bounds.x_min == 3
		bounds.x_max = 40
		...
	"""
	island_bounds = list()
	island_bounds.push(IslandBounds())  # Class 0 -> Nothing.
	island_bounds.push(IslandBounds())  # Class 1 -> Nothing.
	neighborhood = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
	islands = (mat > 0.0).astype(numpy.int)
	# 0 = not a thing.
	# 1 = unlabeled.
	# 2... = unique ID
	latest_id = 2
	for y in range(0, islands.shape[0]):
		for x in range(0, islands.shape[1]):
			if islands[y, x] == untagged_class:
				new_island = IslandBounds(id=latest_id, x_min=x, y_min=y, x_max=x, y_max=y)
				# We have a region heretofore undiscovered.
				pending = [(y, x)]
				while pending:
					nbr_y, nbr_x = pending.pop()
					if islands[nbr_y, nbr_x] == 1:
						islands[nbr_y, nbr_x] = latest_id
						new_island.update_from_coordinate(nbr_x, nbr_y)
						for dy, dx in neighborhood:
							if nbr_y+dy < 0 or nbr_x+dx < 0 or nbr_y+dy >= islands.shape[0] or nbr_x+dx >= islands.shape[1]:
								continue
							if islands[nbr_y+dy, nbr_x+dx] == 1:
								pending.append((nbr_y+dy, nbr_x+dx))
				latest_id += 1
				island_bounds.append(new_island)
	return islands, island_bounds

#
# Workflow
#

def load_image(filename): # Out -> grey image matrix
	"""
	Load an image and convert it to a luminance matrix (float) of the given resolution,
	crop to aspect ratio and normalize to 0/1.

	Returns a numpy array.

	This is a placeholder for the Blender version which will pull a frame from the video stream.
	"""
	from PIL import Image
	img = Image.open(filename).convert('L')
	dst = numpy.asarray(img, dtype=numpy.float32)
	# Normalize:
	dst -= dst.min()
	dst /= dst.max() or 1.0
	return dst

def make_threshold_map(input_matrix):  # Out -> grey image matrix
	# Downscale by four.
	resized = fast_downscale(input_matrix, step=4)
	# Average / blur pixels.
	blurred = blur(resized)
	threshold = resize_linear(blurred, input_matrix.shape[0], input_matrix.shape[1])
	return threshold

def binarize(image_matrix, threshold_map):  # Out -> Image
	return 1.0 * (image_matrix >= threshold_map)

def topological_filter(binarized_image):
	pass

def error_correct(filtered_image):
	pass

#
# Blender interface:
#

def get_animation_frame(idx):
	import bpy

	frameStart = 1
	frameEnd = 155
	frameStep = 50
	viewer_area = None
	viewer_space = None

	for area_search in bpy.context.screen.areas:
		if viewer_area == None and area_search.type == 'IMAGE_EDITOR':
			viewer_area = area_search
			break

	if viewer_area == None:
		viewer_area = bpy.context.screen.areas[0]
		viewer_area.type = "IMAGE_EDITOR"

	for space in viewer_area.spaces:
		if space.type == "IMAGE_EDITOR":
			viewer_space = space

	path = 'H:\\Data\\_blender\\Fluid\\Video_Edit.mov'
	img = bpy.data.images.load(path)
	w = img.size[0]
	h = img.size[1]
	viewer_space.image = img

	frame = 1
	for frame in range(frameStart, frameEnd, frameStep):
		viewer_space.image_user.frame_offset = frame
		# switch back and forth to force refresh
		viewer_space.draw_channels = 'COLOR_ALPHA'
		viewer_space.draw_channels = 'COLOR'
		pixels = list(viewer_space.image.pixels)
		tmp = bpy.data.images.new(name="sample" + str(frame), width=w, height=h, alpha=False, float_buffer=False)
		tmp.pixels = pixels

	img.user_clear()
	bpy.data.images.remove(img)

#
# Helpers:
#

def debug_show(mat):
	from PIL import Image
	img = Image.fromarray(mat*255.0)
	img.show()

def debug_show_islands(classes):
	from PIL import Image
	import itertools
	num_classes = classes.max()
	class_colors = list(itertools.islice(itertools.product(list(range(64, 255, 1)), repeat=3), num_classes+1))
	colored_image = Image.new('RGB', (classes.shape[1], classes.shape[0]))
	# This is the wrong way to do it.  Should just cast + index.
	for y in range(classes.shape[0]):
		for x in range(classes.shape[1]):
			colored_image.putpixel((x,y), class_colors[classes[y,x]])
	colored_image.show()
	return colored_image

def main():
	print("Loading image...")
	img_mat = load_image("test_00.png")
	print("Making threshold map...")
	threshold = make_threshold_map(img_mat)
	print("Binarizing...")
	binarized = binarize(img_mat, threshold)
	print("Selecting islands and connected components...")
	island_pixels, island_data = flood_fill_connected(binarized)
	print("Topo filtering")


#if __name__ == '__main__':
#	main()
