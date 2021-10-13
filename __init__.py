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
if locals().get('topotag_module_loaded'):
	topotag_module_loaded = False  # Set to false until everything is loaded

	print(f"{module_name} already loaded.  Reloading...")
	if module_name in sys.modules:
		import importlib
		sys.modules[module_name] = importlib.reload(module=sys.modules[module_name])
		print("Reloaded")
else:
	print("Loading TopoTagTracker...")

from plugin_main import TopoTagTracker

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

topotag_module_loaded = True