from dataclasses import dataclass, field
from typing import Tuple

import numpy

from image_processing import Matrix


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


def flood_fill_connected(mat) -> Tuple[list, Matrix]:
	"""Takes a black and white matrix with 0 as 'empty' and connect components with value==1.
	Returns a tuple with two items:
	 - A list of length(n+2) where class_n is the position of the bounds information in the list.
	 - int matrix with every pixel assigned to a unique class from 2 to n.

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


def flood_fill_broken(mat) -> Tuple[list, Matrix]:
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
	# First we tag all the positive white/1 islands, then we fill the black/0 empty spaces.
	islands = (mat > 0.0).astype(numpy.int)
	connected_components = _flood_fill_fast(islands, foreground_class=1, starting_class_id=2)
	connected_components_inv = _flood_fill_fast(1-islands, foreground_class=1, starting_class_id=connected_components.max()+1)
	final_components = connected_components + connected_components_inv

	# Collect our island data into useful adjacency info.
	island_bounds = list()
	for i in range(final_components.max()+1):
		island_bounds.append(IslandBounds(id=i))

	# Go back through the final components and find the bounds.
	for y in range(0, final_components.shape[0]):
		for x in range(0, final_components.shape[1]):
			island_bounds[final_components[y, x]].update_from_coordinate(x, y)

	return island_bounds, final_components


def _flood_fill_fast(image: Matrix, foreground_class: int = 1, starting_class_id: int = 1) -> Matrix:
	"""Flood fill the connected components based on the foreground class id."""
	conflict_map = dict()
	last_class_id = starting_class_id
	class_id_used = False
	component_image = numpy.zeros_like(image)
	for y in range(image.shape[0]):
		for x in range(image.shape[1]):
			if image[y,x] == foreground_class:
				# Try to resolve it with the pixel left and above.
				left_class = None
				upper_class = None
				if x-1 >= 0 and image[y, x-1] == foreground_class:
					left_class = component_image[y, x-1]
				if y-1 >= 0 and image[y-1, x] == foreground_class:
					upper_class = component_image[y-1, x]

				if left_class is not None and upper_class is not None:
					# We may have a conflict.
					if left_class != upper_class:
						# We do!  Eventually we'll have to re-assign the max of these classes to be the min.
						conflict_map[max(left_class, upper_class)] = min(left_class, upper_class)
					component_image[y,x] = min(left_class, upper_class)
				elif left_class is not None:
					component_image[y,x] = left_class
				elif upper_class is not None:
					component_image[y,x] = upper_class
				else:
					component_image[y,x] = last_class_id
					class_id_used = True
			else:
				# Now we do not have a connected component.
				if class_id_used:
					# We had assigned a component.
					last_class_id += 1
					class_id_used = False
	# Now that we're done with our scan-line connected component bit, we have to reassign all our conflicts.
	for conf, reassignment in conflict_map.items():
		component_image[component_image == conf] = reassignment

	return component_image