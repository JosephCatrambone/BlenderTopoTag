"""
TopoTag Blender
A plugin to extract topotags from video footage in Blender.
(c) Joseph Catrambone 2021 -- Published under MIT License.
"""

import logging
import numpy

import bgl, bpy

from debug import debug_show_tags
from image_processing import blur, fast_downscale, resize_linear, Matrix
from topotag import find_tags

logger = logging.getLogger(__file__)

bl_info = {
	"name": "Blender TopoTag",
	"blender": (2, 90, 0),
	"category": "Object",
}


class TopoTagTracker(bpy.types.Operator):
	"""Topotag Fiducial Tracking"""
	bl_idname = "object.topotag"
	bl_label = "TopoTag Track"
	bl_options = {"REGISTER", "UNDO"}

	def execute(self, context):
		import pydevd_pycharm
		pydevd_pycharm.settrace('localhost', port=42069, stdoutToServer=True, stderrToServer=True)

		scene = context.scene
		cursor = scene.cursor.location
		obj = object.active_object

		# Push the state to restore user's setup.
		scene_used_nodes = scene.use_nodes
		scene_prev_render_width = scene.render.resolution_x
		scene_prev_render_height = scene.render.resolution_y

		# Set up our scene in a way that lets us render to the composition node and pull pixel data.
		scene.use_nodes = True
		anim_width = bpy.data.movieclips[0].size[0]
		anim_height = bpy.data.movieclips[0].size[1]
		scene.render.resolution_x = anim_width
		scene.render.resolution_y = anim_height

		# Create a preview node so we can directly pull the video data frames.
		tree = scene.node_tree
		nodes = tree.nodes
		links = tree.links

		for node in nodes:
			nodes.remove(node)

		render_layer_node = nodes.new('CompositorNodeRLayers')
		viewer_node = nodes.new('CompositorNodeViewer')
		links.new(viewer_node.inputs[0], render_layer_node.outputs[0])

		# Allocate our empties and perhaps make a collection.
		#bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
		#bpy.ops.object.move_to_collection(collection_index=- 1, is_new=False, new_collection_name='')
		#bpy.ops.object.select_same_collection(collection='')


		for frame_num in range(scene.frame_start, scene.frame_end):
			#currentFrame = scene.frame_current
			#bpy.data.scenes['Scene'].frame_set
			scene.frame_set(frame_num)
			bpy.ops.render.render(write_still=True)
			pixels = numpy.asarray(bpy.data.images['Viewer Node'].pixels)
			image = pixels.reshape((anim_width, anim_height, 4))
			image = convert_pixels(image)

			#

		#buffer = bgl.Buffer(bgl.GL_BYTE, anim_width * anim_height * 4)
		#bgl.glReadPixels(0, 0, anim_width, anim_height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)

		#scene.collection.objects.link(obj_new)
		#obj_new.location = (obj.location * factor) + (cursor * (1.0 - factor))

		# Undo our messing:
		scene.use_nodes = scene_used_nodes
		scene.render.resolution_x = scene_prev_render_width
		scene.render.resolution_y = scene_prev_render_height
		return {'FINISHED'}


def menu_func(self, context):
	self.layout.operator(TopoTagTracker.bl_idname)


def register():
	bpy.utils.register_class(TopoTagTracker)
	bpy.types.CLIP_MT_track.append(TopoTagTracker)


def unregister():
	bpy.utils.unregister_class(TopoTagTracker)

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

def convert_pixels(img) -> Matrix:
	dst = numpy.mean(img, axis=-1)
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
