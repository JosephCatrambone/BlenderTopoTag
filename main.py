"""
TopoTag Blender
A plugin to extract topotags from video footage in Blender.
(c) Joseph Catrambone 2021 -- Published under MIT License.
"""

import logging
import math
import numpy
import sys
from dataclasses import dataclass, field
from typing import Any, NewType, Optional, Tuple, Type

logger = logging.getLogger(__file__)
Matrix = NewType('Matrix', numpy.ndarray)

@dataclass
class RotationMatrix:
	@classmethod
	def to_euler(cls, r):
		"""Return x, y, z such that this (R) = ZYX"""
		x = math.atan2(r[2,1], r[2,2])
		y = math.atan2(-r[2,0], math.sqrt(r[2,1]*r[2,1] + r[2,2]*r[2,2]))
		z = math.atan2(r[1,0], r[0,0])
		return (x, y, z)

	@classmethod
	def from_euler(cls, x: float, y: float, z:float):
		x_rot = cls.from_x_rotation(x)
		y_rot = cls.from_y_rotation(y)
		z_rot = cls.from_z_rotation(z)
		return z_rot @ y_rot @ x_rot

	@classmethod
	def from_x_rotation(cls, val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[1, 0, 0],
			[0, c, -s],
			[0, s, c]
		])

	@classmethod
	def from_y_rotation(cls, val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[c, 0, s],
			[0, 1, 0],
			[-s, 0, c]
		])

	@classmethod
	def from_z_rotation(cls, val: float):
		c = math.cos(val)
		s = math.sin(val)
		return numpy.asarray([
			[c, -s, 0],
			[s, c, 0],
			[0, 0, 1]
		])

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
	z_translation: float

	def to_matrix(self):
		return numpy.hstack([
			RotationMatrix.from_euler(self.x_rotation, self.y_rotation, self.z_rotation),
			numpy.asarray([[self.x_translation, self.y_translation, self.z_translation]]).T
		])

	@classmethod
	def from_naive_dlt(cls, projection, world):
		"""Compute the camera extrinsics from the projected image of the world coordinates."""
		assert projection.shape[1] >= 2
		assert world.shape[1] >= 3
		# s * [u', v', 1].T = [R | t] * [x, y, z, 1].T
		#
		#   0  1  2  3  4  5  6  7    8     9     10   11
		# | x  y  z  1  0  0  0  0  -u'x  -u'y  -u'z  -u' |  * [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz].T = 0
		# | 0  0  0  0  x  y  z  1  -v'x  -v'y  -v'z  -v' |
		# Use SVD to find r|t.
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
		_, _, v = numpy.linalg.svd(homo_mat, full_matrices=False)
		return v[-1,:].reshape((3,4))

	@classmethod
	def from_4point(cls, projection, world):
		"""Apply Yang et. al.'s method from 2009."""
		pass

@dataclass
class IslandBounds:
	id: int = -1
	num_pixels: int = 0
	children: set = field(default_factory=set)  # A set of child IDs, not objects.
	x_min: int = 0
	y_min: int = 0
	x_max: int = 0
	y_max: int = 0

	def __contains__(self, other) -> bool:
		"""Returns True if other (an IslandBounds instance) is entirely inside this."""
		if isinstance(other, IslandBounds):
			if other.x_min <= self.x_min:
				return False
			if other.x_max >= self.x_max:
				return False
			if other.y_min <= self.y_min:
				return False
			if other.y_max >= self.y_max:
				return False
			return True
		else:
			assert len(other) == 2
			return other[0] > self.x_min and other[0] < self.x_max and other[1] > self.y_min and other[1] < self.y_max

	def update_from_coordinate(self, x: int, y: int):
		if self.num_pixels == 0:
			self.x_min = x
			self.y_min = y
			self.x_max = x
			self.y_max = y
		self.num_pixels += 1
		self.x_min = min(self.x_min, x)
		self.y_min = min(self.y_min, y)
		self.x_max = max(self.x_max, x)
		self.y_max = max(self.y_max, y)

	def center(self) -> Tuple[float, float]:
		"""Returns the x,y coordinate of the unweighted center of this rectangle."""
		return (self.x_max+self.x_min)/2, (self.y_max+self.y_min)/2

	def pixel_center(self) -> Tuple[int, int]:
		"""Returns the x,y coordinate of the unweighted center-most pixel of this rectangle."""
		return (self.x_max+self.x_min)//2, (self.y_max+self.y_min)//2

	def max_edge_length(self) -> int:
		"""Return the length of the maximum edge."""
		return max(self.y_max-self.y_min, self.x_max-self.x_min)

	def width(self) -> int:
		return self.x_max - self.x_min

	def height(self) -> int:
		return self.y_max - self.y_min

@dataclass
class TopoTag:
	tag_id: int  # The computed ID of the tag.
	island_id: int  # The raw connected component image has this ID.
	n: int  # The 'order' of the topotag, i.e., the sqrt of the number of internal bits.
	vertex_positions: list  # A list of tuples of x,y, NOT y,x.
	pose: Matrix
	# Useful for debugging and rendering:
	horizontal_baseline: Tuple[float, float] # dx, dy
	vertical_baseline: Tuple[float, float]
	top_left: Tuple[int, int]
	top_right: Tuple[int, int]
	bottom_left: Tuple[int, int]

	@staticmethod
	def generate_points(k: int) -> list:
		"""Returns a list of k^2 coordinates with idx=0 being at 0,0.
		Others match to the coordinates as read in the appropriate vertex order.
		Vertices are ordered left-to-right, top-to-bottom.
		"""
		scale_factor = 1.0
		spacing = scale_factor / float(k-1)
		vertices = list()
		# Fill in the final grid.
		for py in range(0, k):
			for px in range(0, k):
				if py == 0 and px == 1:
					vertices.append((px * (spacing*0.5), py * spacing))
				else:
					vertices.append((px*spacing, py*spacing))
		return vertices

	@staticmethod
	def generate_marker(k: int, code: int, width: int):
		from PIL import Image, ImageDraw
		# We iterate over points in reverse order, so we need to flip the bits in our code.
		bits = list()
		while code > 0:
			if code & 0x1:
				bits.append(1)
			else:
				bits.append(0)
			code //= 2
		bits.append(1)  # For the first two regions.
		bits.append(1)
		if len(bits) > k*k:
			raise Exception(f"Marker with {k}*{k} bits needs {len(bits)} to store {code}")
		bits = list(reversed(bits))

		# Render a color image for the island_matrix.
		img = Image.new('L', size=(width, width))

		padding = width//(2+k)  # The 2+ makes the padding and tags look better.
		work_area_width = width - (2*padding)
		width_per_marker = work_area_width//k  # Reassign, after we remove our border padding.
		pts = TopoTag.generate_points(k)
		canvas = ImageDraw.Draw(img)
		# Fill black, then inner-white to make the rect.
		canvas.rectangle((0, 0, width, width), fill=0)
		canvas.rectangle((padding//4, padding//4, width-(padding//4), width-(padding//4)), fill=255)
		# Draw all the points.
		for bit_id, p in enumerate(pts):
			x = (p[0] * work_area_width)
			y = (p[1] * work_area_width)
			canvas.ellipse((
				x-(width_per_marker//2)+padding, y-(width_per_marker//2)+padding,
				x+(width_per_marker//2)+padding, y+(width_per_marker//2)+padding
				), fill=0
			)
			if bit_id < len(bits) and bits[bit_id]:
				canvas.ellipse((
					x - (width_per_marker // 4) + padding, y - (width_per_marker // 4) + padding,
					x + (width_per_marker // 4) + padding, y + (width_per_marker // 4) + padding
					), fill=255
				)
				canvas.text((x+padding, y+padding), f"Bit {bit_id} {bits[bit_id]}", fill=120)
		return img

	@staticmethod
	def from_island_data(island_id, island_data: list, camera_intrinsics: Optional[CameraIntrinsics] = None) -> Optional:
		"""Given the ID of an island to decode, the list of all island data, and the matrix of connected components,
		attempt to decode the island with the given ID into a TopoTag.  Will return a TopoTag or None."""

		# TODO: This should filter pixels which are less than a certain amount of the baseline area.

		# Quick reject regions too small:
		if island_data[island_id].num_pixels < 10*10 or island_data[island_id].width() < 16 or island_data[island_id].height() < 16:
			return None

		# The first pixel of each region is labeled with a capital letter.
		# island_id in this case will be 'A' (though it's actually an int)
		# When we locate the first region, it will be ID B.
		# We expect that all children of A, [B, E, G, H] to have one or zero children. (Except B.)
		#
		# ##################
		# #A              #
		# # B####     E## #
		# # #C#D#     #F# #
		# # #####     ### #
		# #               #
		# # G##       H## #
		# # ###       #I# #
		# # ###       ### #
		# #               #
		# #################
		#

		# NOTE: We do not re-evaluate the 'contains' of the children.  We assume that they're all 'inside'.

		# For efficiency, we do two things in this pass:
		# - Find the 'first' region -- the one with exactly two children.
		# - While we're at it, make sure that none of the child regions have a depth of more than one.
		#   Each child needs exactly one or zero children.
		baseline_region_id = None
		for child_id in island_data[island_id].children:  # Should be n^2 children...
			if len(island_data[child_id].children) == 2:  # This black region, if it's the first, needs two children.
				# This corresponds to 'B' in our diagram above.
				if baseline_region_id is not None:
					# We already found a first region, so this means there's more than one and this is not a valid tag.
					return None
				baseline_region_id = child_id  # Otherwise, valid candidate.
			# A child node other than the first can have one or zero children. E, G, or H in our diagram.
			if len(island_data[child_id].children) > 2:
				return None  # Bad tag.
			# We should also check here that the child has no children, but...
		if baseline_region_id is None:
			# No region detected -- not a valid tag.
			return None

		# For an added layer of sanity, the depth of this tree must be _at max_ 3.
		for child_id in island_data[island_id].children:
			for grandchild_id in island_data[child_id].children:
				if len(island_data[grandchild_id].children) != 0:
					return None

		# Find third region.  (E in our diagram above.)
		# The two grand children in first region (C & D) define a direction to search for the third region.
		grandchildren_ids = list(island_data[baseline_region_id].children)
		first_region_id = grandchildren_ids[0]
		second_region_id = grandchildren_ids[1]
		first_region_center = island_data[first_region_id].center()
		second_region_center = island_data[second_region_id].center()
		# The third region is the farthest away from 1&2 that's still inside AND on the line defined by 1-2.
		# Any child of this region is, by definition, inside, so we need only to verify the pixels are on the line.
		dx, dy = second_region_center[0] - first_region_center[0], second_region_center[1] - first_region_center[1]
		baseline_horizontal_regions = find_regions_along_line(first_region_center, (dx, dy), island_id, island_data)
		if len(baseline_horizontal_regions) == 0:
			# print("No 3rd region found")
			return None
		third_region_id = baseline_horizontal_regions[-1]
		third_region_center = island_data[third_region_id].center()
		baseline_horizontal_slope = (third_region_center[0] - first_region_center[0], third_region_center[1] - first_region_center[1])
		baseline_angle = math.atan2(baseline_horizontal_slope[1], baseline_horizontal_slope[0])

		# Now we actually can pick the true 'first' region in the paper.  Region B in our diagram.
		# If region two is farther region three than region one, swap one and two.
		if (abs(second_region_center[0]-third_region_center[0])+abs(second_region_center[1]-third_region_center[1])) > (abs(first_region_center[0]-third_region_center[0])+abs(first_region_center[1]-third_region_center[1])):
			first_region_id, second_region_id = second_region_id, first_region_id
			first_region_center, second_region_center = second_region_center, first_region_center

		# Now that we have region 2 and 3, use that to find 4.
		dx, dy = third_region_center[0] - first_region_center[0], third_region_center[1] - first_region_center[1]
		# A dot B / mag(a)*mag(b) = cos theta
		# The forth region is the one that makes the biggest angle with respect to one and three.
		max_angle_to_r4 = 0
		baseline_vertical_slope = (-dy, dx)  # Sorta' hack: assume 90 degree, but this will be overridden below.
		for candidate in island_data[island_id].children:
			# Calculate the angle.
			if candidate == first_region_id:
				continue
			forth_region_center = island_data[candidate].center()
			candidate_slope = (forth_region_center[0] - first_region_center[0], forth_region_center[1] - first_region_center[1])
			angle = math.fabs(math.atan2(candidate_slope[1], candidate_slope[0])-baseline_angle)
			if angle > max_angle_to_r4:
				max_angle_to_r4 = angle
				baseline_vertical_slope = candidate_slope
		baseline_vertical_regions = find_regions_along_line(first_region_center, baseline_vertical_slope, island_id, island_data)
		if len(baseline_vertical_regions) == 0:
			# print("Can't find 4th region")
			return None
		forth_region_id = baseline_vertical_regions[-1]
		forth_region_center = island_data[forth_region_id].center()

		# Finally, decode our tag and get the vertex positions.
		all_region_ids = {baseline_region_id, first_region_id, second_region_id}
		vertices = [first_region_center, second_region_center]
		# The corners are locked.  We can use our horizontal and vertical baselines to read 'left to right' the untouched islands.
		for vertical_region in baseline_vertical_regions:  # Top to bottom.
			for horizontal_region in find_regions_along_line(island_data[vertical_region].center(), baseline_horizontal_slope, island_id, island_data):
				if horizontal_region in island_data[island_id].children and horizontal_region not in all_region_ids:
					vertices.append(island_data[horizontal_region].center())
					all_region_ids.add(horizontal_region)
		# Decode the regions.
		code = 0
		for (bit_id, region) in enumerate(all_region_ids):
			if bit_id == 0 or bit_id == 1 or bit_id == 2:  # Skip baseline region, first region, and second.
				continue
			if len(island_data[region].children) > 0:
				code += 1
			code = code << 1
		code = code >> 1  # We have an extra divide-by-two.

		# Recover pose from marker positions.
		# Start by computing the fake 'planar' case where we assume island 0 is at 0,0,0 in the world.
		# Basically assume that the marker region 1 is at 0,0 and move right.
		k_value = int(math.sqrt(len(all_region_ids)))
		if k_value < 3:
			return None
		positions_2d = numpy.asarray(TopoTag.generate_points(k_value))
		positions_3d = numpy.hstack([positions_2d, numpy.ones(shape=(positions_2d.shape[0], 1))])
		# pos_2d is our 'projection'.  Pretend it exists at the origin in R3.
		projection = CameraExtrinsics.from_naive_dlt(numpy.asarray(vertices), positions_3d)

		result = TopoTag(
			code,
			island_id,
			k_value,
			vertices,
			projection,
			baseline_horizontal_slope,
			baseline_vertical_slope,
			top_left=first_region_center,
			top_right=third_region_center,
			bottom_left=forth_region_center
		)
		return result


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

def find_regions_along_line(origin: Tuple[float, float], dxdy: Tuple[float, float], island_id:int, island_data:list) -> list:
	"""Returns a list of the region IDs on the given line inside the island.  Sorted by increasing distance from origin."""
	# This is a dumb and lazy way to do it, but we can move along the simplified line defined by dx/dy.
	dx = dxdy[0] / max(abs(dxdy[0]), abs(dxdy[1]))
	dy = dxdy[1] / max(abs(dxdy[0]), abs(dxdy[1]))
	print(f"Searching from {origin[0],origin[1]} along line {dx},{dy}")
	# Keep in mind that this marker _isn't_ perspective aligned at this point so we can't make any assumptions.
	max_steps = island_data[island_id].max_edge_length()
	regions_on_line = set()
	for step in range(-max_steps, max_steps):
		xy = (origin[0] + (dx * step), origin[1] + (dy * step))
		# if x < 0 or x > island_matrix.shape[1] or y < 0 or y > island_matrix.shape[0]:
		if xy in island_data[island_id]:
			for child_id in island_data[island_id].children:
				if xy in island_data[child_id]:
					regions_on_line.add(child_id)
	# Convert the set to a list and sort:
	return sorted(regions_on_line, key=lambda pt: abs(island_data[pt].center()[0]-origin[0])+abs(island_data[pt].center()[1]-origin[1]))

def flood_fill_connected(mat) -> Tuple[Matrix, list]:
	"""Takes a black and white matrix with 0 as 'empty' and connect components with value==1.
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
	island_bounds.append(IslandBounds())  # Class 0 -> Nothing.
	island_bounds.append(IslandBounds())  # Class 1 -> Nothing.
	neighborhood = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
	islands = (mat > 0.0).astype(numpy.int)

	latest_id = 2
	# First we tag all the positive white/1 islands, then we fill the black/0 empty spaces.
	for untagged_class in [1, 0]: # THIS MUST BE 1, 0.
		for y in range(0, islands.shape[0]):
			for x in range(0, islands.shape[1]):
				if islands[y, x] == untagged_class:
					new_island = IslandBounds(id=latest_id, x_min=x, y_min=y, x_max=x, y_max=y)
					# We have a region heretofore undiscovered.
					pending = [(y, x)]
					while pending:
						nbr_y, nbr_x = pending.pop()
						if islands[nbr_y, nbr_x] == untagged_class:
							islands[nbr_y, nbr_x] = latest_id
							new_island.update_from_coordinate(nbr_x, nbr_y)
							for dy, dx in neighborhood:
								if nbr_y+dy < 0 or nbr_x+dx < 0 or nbr_y+dy >= islands.shape[0] or nbr_x+dx >= islands.shape[1]:
									continue
								if islands[nbr_y+dy, nbr_x+dx] == untagged_class:
									pending.append((nbr_y+dy, nbr_x+dx))
					latest_id += 1
					island_bounds.append(new_island)
	return island_bounds, islands

#
# Workflow
#

def load_image(filename) -> Matrix: # -> grey image matrix
	"""
	Load an image and convert it to a luminance matrix (float) of the given resolution,
	crop to aspect ratio and normalize to 0/1.

	Returns a numpy array.

	This is a placeholder for the Blender version which will pull a frame from the video stream.
	"""
	from PIL import Image
	img = Image.open(filename)
	return convert_image(img)

def convert_image(img) -> Matrix:
	"""Perform the required conversion and preprocessing on an image object."""
	dst = numpy.asarray(img.convert('L'), dtype=numpy.float32)
	# Normalize:
	dst -= dst.min()
	dst /= dst.max() or 1.0
	return dst

def make_threshold_map(input_matrix: Matrix) -> Matrix:  # -> grey image matrix
	"""This is basically just blur."""
	# Downscale by four.
	resized = fast_downscale(input_matrix, step=4)
	# Average / blur pixels.
	blurred = blur(resized)
	threshold = resize_linear(blurred, input_matrix.shape[0], input_matrix.shape[1])*0.5
	return threshold

def binarize(image_matrix: Matrix) -> Matrix:
	"""Return a binary integer matrix with ones and zeros."""
	# Should we just combine this with the make_threshold_map function?
	#threshold_map = make_threshold_map(image_matrix)
	#return (image_matrix >= threshold_map).astype(int)
	return (image_matrix > image_matrix.mean()+image_matrix.std()*0.5).astype(int)

def find_tags(binarized_image: Matrix) -> (list, list, Matrix):
	"""Given the binarized image data, return a tuple of the topotags, the island data, the connected component matrix."""
	island_data, island_matrix = flood_fill_connected(binarized_image)

	# We have a bunch of unconnected (flat) island data.
	# Our data structure is built left-to-right because it's faster to access memory in that order,
	# but this means our rectangles are probably in the wrong order.
	# We have to sort them or iterate over them in them top-to-bottom, but this leaves more in the 'open set'.
	# Or...

	# TODO: There's a stupid O(n^2) and then there's a REALLY stupid _this_.
	# Iterate over the island_matrix and get the pairs of touching components, then build the hierarchy from that.
	touching_pairs = set()
	for y in range(island_matrix.shape[0]-1):
		for x in range(island_matrix.shape[1]-1):
			v = island_matrix[y, x]
			vdx = island_matrix[y, x+1]
			vdy = island_matrix[y+1, x]
			if v != vdx:
				touching_pairs.add((min(vdx, v), max(vdx, v)))
			if v != vdy:
				touching_pairs.add((min(vdy, v), max(vdy, v)))

	# Now go over all pairs and, if one is a child of another, add it.
	for a,b in touching_pairs:
		if island_data[a] in island_data[b]:
			island_data[b].children.add(a)
		if island_data[b] in island_data[a]:
			island_data[a].children.add(b)

	topo_tags = list()
	for island_id in range(2, len(island_data)):
		tag = TopoTag.from_island_data(island_id, island_data)
		if tag:
			topo_tags.append(tag)
	return topo_tags, island_data, island_matrix

#
# Blender interface:
#

def get_animation_frame(idx):
	"""Load video frame hack from S.O.  TODO: Credit author in title."""
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

def debug_show_islands(classes, show=True):
	from PIL import Image
	import itertools
	num_classes = classes.max()
	class_colors = list(itertools.islice(itertools.product(list(range(64, 255, 1)), repeat=3), num_classes+1))
	colored_image = Image.new('RGB', (classes.shape[1], classes.shape[0]))
	# This is the wrong way to do it.  Should just cast + index.
	for y in range(classes.shape[0]):
		for x in range(classes.shape[1]):
			colored_image.putpixel((x,y), class_colors[classes[y,x]])
	if show:
		colored_image.show()
	return colored_image

def debug_show_tags(tags, island_data, island_matrix, show=True):
	from PIL import Image, ImageDraw
	# Render a color image for the island_matrix.
	img = debug_show_islands(island_matrix, show=False)
	canvas = ImageDraw.Draw(img)
	# Draw some red borders for candidate islands.
	#for island in island_data[2:]:
	#	canvas.rectangle((island.x_min, island.y_min, island.x_max, island.y_max), outline=(255, 0, 0))
	# Draw a pink border for each tag.
	for tag in tags:
		island_id = tag.island_id
		for vertex in tag.vertex_positions:
			canvas.rectangle((vertex[0]-1, vertex[1]-1, vertex[0]+1, vertex[1]+1), outline=(255, 255, 255))
		canvas.text((island_data[island_id].x_min, island_data[island_id].y_min), f"I{island_id} - Code{tag.tag_id}", fill=(255, 255, 255))
		canvas.rectangle((island_data[island_id].x_min, island_data[island_id].y_min, island_data[island_id].x_max, island_data[island_id].y_max), outline=(255, 0, 255))
		canvas.line((tag.top_left[0], tag.top_left[1], tag.top_right[0], tag.top_right[1]), fill=(0, 255, 255))
		canvas.line((tag.top_left[0], tag.top_left[1], tag.bottom_left[0], tag.bottom_left[1]), fill=(0, 255, 255))

	if show:
		img.show()
	return img

def main(image_filename: str = None, image = None):
	print("Loading image...")
	if image_filename:
		img_mat = load_image(image_filename)
	elif image:
		img_mat = convert_image(image)
	print("Binarizing...")
	binary_mat = binarize(img_mat)
	print("Finding tags...")
	tags, island_data, island_pixels = find_tags(binary_mat)
	for t in tags:
		print(t)
	debug_show_tags(tags, island_data, island_pixels)

#if __name__ == '__main__':
#	main(sys.argv[1])
