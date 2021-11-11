import numpy

from fiducial import TopoTag


def save_plain_ppm(img, filename: str):
	"""Save a matrix (floats) as a PPM image."""
	with open(filename, 'wt') as fout:
		fout.write("P3\n")
		fout.write(f"{img.shape[1]} {img.shape[0]}\n")
		fout.write("255\n")
		idx = 0
		for y in range(img.shape[0]):
			for x in range(img.shape[1]):
				if len(img.shape) == 2:
					fout.write(str(int(255 * img[y, x])))
					fout.write(" ")
					fout.write(str(int(255 * img[y, x])))
					fout.write(" ")
					fout.write(str(int(255 * img[y, x])))
				elif len(img.shape) == 3:
					fout.write(str(int(255 * img[y, x, 0])))
					fout.write(" ")
					fout.write(str(int(255 * img[y, x, 1])))
					fout.write(" ")
					fout.write(str(int(255 * img[y, x, 2])))

				if idx >= 5:  # Max line length is 70. 3 digits + space * 3 channels -> 12.  70/12 ~> 5.
					fout.write("\n")
					idx = 0
				else:
					fout.write(" ")
					idx += 1
		fout.flush()

def debug_render_cube(tag: TopoTag, canvas):
	"""Render a cube from the perspective of the camera."""
	points_3d = numpy.asarray([
		[0, 0, 0, 1],
		[1, 0, 0, 1],
		[1, 1, 0, 1],
		[0, 1, 0, 1],
		[0, 0, 1, 1],
		[1, 0, 1, 1],
		[1, 1, 1, 1],
		[0, 1, 1, 1],
	])
	projection_matrix = tag.pose_raw # tag.extrinsics.to_matrix()
	projection = (projection_matrix @ points_3d.T).T
	projection[:, 0] /= projection[:, 2]
	projection[:, 1] /= projection[:, 2]
	#projection[:, 2] /= projection[:, 2]
	# Draw faces...
	for i in range(0, 4):
		canvas.line((projection[i, 0], projection[i, 1], projection[(i+1)%4, 0], projection[(i+1)%4, 1]), fill=(255, 255, 255))
		canvas.line((projection[(i+4), 0], projection[(i+4), 1], projection[(i+5)%8, 0], projection[(i+5)%8, 1]), fill=(255, 255, 255))
	# Draw edges between faces (for the other faces)
	print(projection)


def debug_show_tags(tags, island_data, island_matrix, show=True):
	from PIL import ImageDraw
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
			canvas.rectangle((vertex[0]-1, vertex[1]-1, vertex[0]+1, vertex[1]+1), outline=(200, 200, 200))
		canvas.text((island_data[island_id].x_min, island_data[island_id].y_min), f"I{island_id} - Code{tag.tag_id}", fill=(255, 255, 255))
		canvas.rectangle((island_data[island_id].x_min, island_data[island_id].y_min, island_data[island_id].x_max, island_data[island_id].y_max), outline=(255, 0, 255))
		canvas.line((tag.top_left[0], tag.top_left[1], tag.top_right[0], tag.top_right[1]), fill=(0, 255, 255))
		canvas.line((tag.top_left[0], tag.top_left[1], tag.bottom_left[0], tag.bottom_left[1]), fill=(0, 255, 255))
		#debug_render_cube(tag, canvas)
		print(f"Tag origin: {tag.extrinsics.x_translation}, {tag.extrinsics.y_translation}, {tag.extrinsics.z_translation}")

	if show:
		img.show()
	return img


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


def debug_show(mat):
	from PIL import Image
	img = Image.fromarray(mat*255.0)
	img.show()