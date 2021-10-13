import math
from dataclasses import dataclass
from typing import List, Type, Tuple, Optional

import numpy

from camera import CameraIntrinsics, CameraExtrinsics
from computer_vision import calibrate_camera_from_known_points, refine_camera
from island import flood_fill_connected
from image_processing import Matrix


@dataclass
class TopoTag:
	tag_id: int  # The computed ID of the tag.
	island_id: int  # The raw connected component image has this ID.
	n: int  # The 'order' of the topotag, i.e., the sqrt of the number of internal bits.
	vertex_positions: list  # A list of tuples of x,y, NOT y,x.
	intrinsics: Type[CameraIntrinsics]
	extrinsics: Type[CameraExtrinsics]
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
		bits = [False] * (k*k)
		for bit_idx in range(2, len(bits)):
			if code & (0x1 << (bit_idx-2)):
				bits[bit_idx] = True
			else:
				bits[bit_idx] = False
		if math.ceil(math.log2(max(1,code))) > k*k:
			raise Exception(f"Marker with {k}*{k} bits needs {len(bits)} to store {code} (with hold-out of 2)")
		bits.append(True)  # For the first two regions.
		bits.append(True)
		bits = list(reversed(bits))  # This feels completely backwards from what the paper says, but matches their code.

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
				#canvas.text((x+padding, y+padding), f"Bit {bit_id} {bits[bit_id]}", fill=120)
		return img

	@staticmethod
	def from_island_data(island_id, island_data: list, camera_intrinsics: Optional[CameraIntrinsics] = None) -> Optional:
		"""Given the ID of an island to decode, the list of all island data, and the matrix of connected components,
		attempt to decode the island with the given ID into a TopoTag.  Will return a TopoTag or None."""

		# Quick reject regions too small:
		if island_data[island_id].num_pixels < 10*10 or island_data[island_id].width() < 16 or island_data[island_id].height() < 16:
			return None

		# We should do better to estimate the camera matrix.
		if camera_intrinsics is None:
			camera_intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 0, 0)

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
		positions_3d = numpy.hstack([positions_2d, numpy.zeros(shape=(positions_2d.shape[0], 1))])# @ numpy.linalg.inv(camera_intrinsics.to_matrix())
		# pos_2d is our 'projection'.  Pretend it exists at the origin in R3.
		intrinsics, extrinsics = calibrate_camera_from_known_points(numpy.asarray(vertices), positions_3d)
		intrinsics, extrinsics = refine_camera(positions_2d, positions_3d, intrinsics, extrinsics)

		result = TopoTag(
			code,
			island_id,
			k_value,
			vertices,
			intrinsics,
			extrinsics,
			baseline_horizontal_slope,
			baseline_vertical_slope,
			top_left=first_region_center,
			top_right=third_region_center,
			bottom_left=forth_region_center
		)
		return result


def find_tags(image: Matrix) -> (List[Type[TopoTag]], list, Matrix):
	"""Given a greyscale image matrix, return a tuple of (topotags, island data, connected component matrix)."""
	binarized_image = binarize(image)
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

	# Ballpark camera intrinsics for now.
	camera_intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, binarized_image.shape[1] // 2, binarized_image.shape[0] // 2)

	topo_tags = list()
	for island_id in range(2, len(island_data)):
		tag = TopoTag.from_island_data(island_id, island_data, camera_intrinsics)
		if tag:
			topo_tags.append(tag)
	return topo_tags, island_data, island_matrix


def binarize(image_matrix: Matrix) -> Matrix:
	"""Return a binary integer matrix with ones and zeros."""
	# Should we just combine this with the make_threshold_map function?
	#threshold_map = make_threshold_map(image_matrix)
	#return (image_matrix >= threshold_map).astype(int)
	return (image_matrix > image_matrix.mean()+image_matrix.std()*0.5).astype(int)


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