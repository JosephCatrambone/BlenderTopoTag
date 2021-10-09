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