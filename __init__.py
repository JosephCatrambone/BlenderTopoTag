bl_info = {
	"name": "Blender TopoTag",
	"blender": (2, 90, 0),
	"category": "Video Tools",
}

import sys, os

# Plugin hacks to make the add-on register:
path = os.path.dirname(__file__)
if path not in sys.path:
	sys.path.append(path)

if __name__ != "__main__":
	module_name = __name__
else:
	module_name = os.path.split(path)[-1]

# The magic incantation to force-reload to operate properly:
if module_name in sys.modules and locals().get('plugin_main') is not None:
	print(f"{module_name} already loaded.  Reloading...")
	import importlib
	importlib.reload(plugin_main)
	#sys.modules[module_name] = importlib.reload(module=sys.modules[module_name])
	print("Reloaded")
else:
	print("Loading TopoTagTracker...")
	import plugin_main

import bpy


def menu_func(self, context):
	self.layout.operator(plugin_main.TopoTagTracker.bl_idname)


def register():
	bpy.utils.register_class(plugin_main.TopoTagTracker)
	bpy.types.CLIP_MT_track.append(plugin_main.TopoTagTracker)


def unregister():
	bpy.utils.unregister_class(plugin_main.TopoTagTracker)


if __name__== "__main__":
	register()