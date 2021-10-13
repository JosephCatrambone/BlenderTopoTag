bl_info = {
	"name": "Blender TopoTag",
	"blender": (2, 90, 0),
	"category": "Video Tools",
}

import sys
if __name__ not in sys.modules:
	sys.modules['TopoTag'] =

# The magic incantation to force-reload to operate properly:
if "bpy" in locals():
	import importlib
	importlib.reload(module=main)
	print("Reloaded")
else:
	from . main import TopoTagTracker

import bpy


def menu_func(self, context):
	self.layout.operator(TopoTagTracker.bl_idname)


def register():
	bpy.utils.register_class(TopoTagTracker)
	bpy.types.CLIP_MT_track.append(TopoTagTracker)


def unregister():
	bpy.utils.unregister_class(TopoTagTracker)


if __name__== "__main__":
	register()