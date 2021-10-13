"""
TopoTag Blender
A plugin to extract topotags from video footage in Blender.
(c) Joseph Catrambone 2021 -- Published under MIT License.
"""

import logging
import numpy

import bpy

from debug import debug_show_tags
from image_processing import blur, fast_downscale, resize_linear, Matrix
from topotag import find_tags

logger = logging.getLogger(__file__)


class TopoTagTracker(bpy.types.Operator):
	"""Topotag Fiducial Tracking"""
	bl_idname = "tracking.track_topotags"
	bl_label = "Track TopoTags"
	bl_options = {"UNDO"}

	tag_width: bpy.props.FloatProperty()

	def __init__(self, *args, **kwargs):
		super(TopoTagTracker, self).__init__(*args, **kwargs)
		self.fiducial_objects = dict()
		self.context = None

	def create_or_fetch_fiducial(self, fid):
		# Create and link a new fiducial in the fiducial collection if it exists OR create it and the collection.
		if fid in self.fiducial_objects:
			return self.fiducial_objects[fid]

		empty_data = bpy.data.objects.new(f"Fiducial_{fid}", None)
		empty_data.empty_display_size = 2
		empty_data.empty_display_type = 'PLAIN_AXES'
		bpy.context.scene.collection.objects.link(empty_data)
		#self.context.view_layer.active_layer_collection.collection.objects.link(light_object)
		#light_object.select_set(True)
		#view_layer.objects.active = light_object
		self.fiducial_objects[fid] = empty_data

		return empty_data

	def invoke(self, context, event):
		#self.marker_size = event.mouse_y
		#self.tag_width =
		return self.execute(context)

	def execute(self, context):
		#import pydevd_pycharm
		#pydevd_pycharm.settrace('localhost', port=42069, stdoutToServer=True, stderrToServer=True)
		self.context = context
		scene = context.scene

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
		# This will create the object and make it active but not return a reference.
		#bpy.ops.object.empty_add(type='CUBE', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

		#bpy.ops.object.move_to_collection(collection_index=- 1, is_new=False, new_collection_name='')
		#bpy.ops.object.select_same_collection(collection='')

		for frame_num in range(scene.frame_start, scene.frame_end):
			#currentFrame = scene.frame_current
			#bpy.data.scenes['Scene'].frame_set
			scene.frame_set(frame_num)
			# This will NOT work when run in the background.
			# https://blender.stackexchange.com/questions/69230/python-render-script-different-outcome-when-run-in-background/81240#81240
			bpy.ops.render.render(write_still=True)
			pixels = numpy.asarray(bpy.data.images['Viewer Node'].pixels)
			# buffer = bgl.Buffer(bgl.GL_BYTE, anim_width * anim_height * 4)
			# bgl.glReadPixels(0, 0, anim_width, anim_height, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)
			image = pixels.reshape((anim_width, anim_height, 4))
			image = convert_pixels(image)
			tags, _, _ = find_tags(image)
			for tag in tags:
				scene_tag = self.create_or_fetch_fiducial(fid=tag.tag_id)
				scene_tag.location = (-tag.extrinsics.x_translation, -tag.extrinsics.y_translation, -tag.extrinsics.z_translation)
				scene_tag.rotation_euler = (-tag.extrinsics.x_rotation, -tag.extrinsics.y_rotation, -tag.extrinsics.z_rotation)
				scene_tag.keyframe_insert(data_path="location", frame=frame_num)
				scene_tag.keyframe_insert(data_path="rotation", frame=frame_num)
		#scene.collection.objects.link(obj_new)

		# Undo our messing:
		scene.use_nodes = scene_used_nodes
		scene.render.resolution_x = scene_prev_render_width
		scene.render.resolution_y = scene_prev_render_height
		return {'FINISHED'}


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


if __name__ == '__main__':
	#main(sys.argv[1])
	register()
