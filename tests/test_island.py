import pprofile
from datetime import datetime

import numpy

from island import flood_fill_connected


perf_benchmarks = """
v0:
    66|        10|            0|            0|  0.00%|def flood_fill_connected(mat) -> Tuple[Matrix, list]:
    79|        10|            0|            0|  0.00%|	island_bounds = list()
    80|        10|            0|            0|  0.00%|	island_bounds.append(IslandBounds())  # Class 0 -> Nothing.
(call)|        10|            0|            0|  0.00%|# E:\PythonProjects\blender_topotag\island.py:2 __init__
    81|        10|            0|            0|  0.00%|	island_bounds.append(IslandBounds())  # Class 1 -> Nothing.
(call)|        10|            0|            0|  0.00%|# E:\PythonProjects\blender_topotag\island.py:2 __init__
    82|        10|            0|            0|  0.00%|	neighborhood = [(-1, 0), (+1, 0), (0, -1), (0, +1)]
    83|        10|    0.0333533|   0.00333533|  0.00%|	islands = (mat > 0.0).astype(numpy.int)
(call)|        10|   0.00401878|  0.000401878|  0.00%|# E:\Applications\miniconda3\envs\blender_topotag\lib\site-packages\numpy\__init__.py:276 __getattr__
    84|         0|            0|            0|  0.00%|
    85|        10|            0|            0|  0.00%|	latest_id = 2
    86|         0|            0|            0|  0.00%|	# First we tag all the positive white/1 islands, then we fill the black/0 empty spaces.
    87|        30|            0|            0|  0.00%|	for untagged_class in [1, 0]: # THIS MUST BE 1, 0.
    88|     24520|    0.0450435|  1.83701e-06|  0.00%|		for y in range(0, islands.shape[0]):
    89|  40349500|      69.5448|  1.72356e-06|  5.10%|			for x in range(0, islands.shape[1]):
    90|  40325000|      80.4508|  1.99506e-06|  5.90%|				if islands[y, x] == untagged_class:
    91|        10|            0|            0|  0.00%|					new_island = IslandBounds(id=latest_id, x_min=x, y_min=y, x_max=x, y_max=y)
(call)|        10|            0|            0|  0.00%|# E:\PythonProjects\blender_topotag\island.py:2 __init__
    92|         0|            0|            0|  0.00%|					# We have a region heretofore undiscovered.
    93|        10|            0|            0|  0.00%|					pending = [(y, x)]
    94|  40300520|      69.2826|  1.71915e-06|  5.08%|					while pending:
    95|  40300510|      75.6694|  1.87763e-06|  5.54%|						nbr_y, nbr_x = pending.pop()
    96|  40300510|      82.3602|  2.04365e-06|  6.04%|						if islands[nbr_y, nbr_x] == untagged_class:
    97|  20162500|      39.6767|  1.96785e-06|  2.91%|							islands[nbr_y, nbr_x] = latest_id
    98|  20162500|      100.863|  5.00248e-06|  7.39%|							new_island.update_from_coordinate(nbr_x, nbr_y)
(call)|  20162500|      215.457|   1.0686e-05| 15.79%|# E:\PythonProjects\blender_topotag\island.py:35 update_from_coordinate
    99| 100812500|      182.512|  1.81041e-06| 13.37%|							for dy, dx in neighborhood:
   100|  80650000|      187.892|  2.32972e-06| 13.77%|								if nbr_y+dy < 0 or nbr_x+dx < 0 or nbr_y+dy >= islands.shape[0] or nbr_x+dx >= islands.shape[1]:
   101|     12250|    0.0170262|  1.38989e-06|  0.00%|									continue
   102|  80601000|      179.194|  2.22322e-06| 13.13%|								if islands[nbr_y+dy, nbr_x+dx] == untagged_class:
   103|  40300500|      81.5021|  2.02236e-06|  5.97%|									pending.append((nbr_y+dy, nbr_x+dx))
   104|        10|            0|            0|  0.00%|					latest_id += 1
   105|        10|            0|            0|  0.00%|					island_bounds.append(new_island)
   106|        10|            0|            0|  0.00%|	return island_bounds, islands



"""


def run_connected_component_performance_bench():
	for i in range(100, 2600, 250):
		mat = numpy.random.uniform(size=(i,i)) > 0.0
		mat = mat.astype(numpy.uint8)
		_ = flood_fill_connected(mat)


def test_connected_component_runtime():
	#cProfile.runctx('run_connected_component_performance_bench()', globals(), locals(), filename=None)
	profiler = pprofile.Profile()
	with profiler:
		run_connected_component_performance_bench()
	profiler.dump_stats(f"{str(datetime.utcnow()).replace(':', '').replace('.', '').replace('-', '')}_connected_component_profile.txt")
	profiler.print_stats()