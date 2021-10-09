"""
TopoTag Blender
A plugin to extract topotags from video footage in Blender.
(c) Joseph Catrambone 2021 -- Published under MIT License.
"""

import logging
import numpy

from image_processing import blur, fast_downscale, resize_linear, Matrix
from topotag import TopoTag, find_tags

logger = logging.getLogger(__file__)


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
	threshold = resize_linear(blurred, input_matrix.shape[0], input_matrix.shape[1]) * 0.5
	return threshold


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

def main(image_filename: str = None, image = None):
	print("Loading image...")
	if image_filename:
		img_mat = load_image(image_filename)
	elif image:
		img_mat = convert_image(image)
	print("Finding tags...")
	tags, island_data, island_pixels = find_tags(img_mat)
	for t in tags:
		print(t)
	debug_show_tags(tags, island_data, island_pixels)

#if __name__ == '__main__':
#	main(sys.argv[1])
