# import numba
# import cv2
# from PIL import Image
# import numpy as np
# import heapq
#
# START = (551, 486)
# END = (1435, 2469)
# TGT_A = (791, 1441)
# TGT_B = (982, 1551)
# TGT_C = (1410, 2181)
# sqrt2 = np.sqrt(2)
#
#
# @numba.njit
# def neighbour_coords(coords, map_array):
#     nbs = []
#     for i in range(-1, 2):
#         for j in range(-1, 2):
#             if not (i == 0 and j == 0):
#                 nbr = (coords[0] + i, coords[1] + j)
#                 if 0 <= nbr[0] < map_array.shape[0] and 0 <= nbr[1] < map_array.shape[1]:
#                     if map_array[nbr] != 0:
#                         nbs.append(nbr)
#     return nbs
#
#
# @numba.njit(fastmath=True)
# def euclidian_distance(p1, p2):
#     return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
#
#
# # @numba.njit
# # def a_star(start, end, map_array):
# #     shape = map_array.shape
# #     g_values = np.full(shape, np.inf, dtype=np.float32)          # Array to store values for g(n)
# #     h_values = np.full(shape, np.inf, dtype=np.float32)          # Array to store values for h(n)
# #     # prev = np.full(shape + (2,), -1, dtype=np.int32)            # Array to store parent node
# #     prev = np.full(shape, (-1, -1))
# #     closed_set = [start]                                        # List of explored nodes
# #     closed_set.pop(0)
# #     open_set = [(euclidian_distance(start, end), start)]
# #     heapq.heapify(open_set)
# #     g_values[start] = 0                                         # Initialise with start point coords and values
# #     h_values[start] = euclidian_distance(start, end)
# #     q_map = dict()
# #     start_f_value = g_values[start] + h_values[start]
# #     q_map[start] = (g_values[start] + h_values[start], start)
# #
# #     def add_item(item: tuple[float, tuple[int, int]], priority: float):
# #         if item in q_map:
# #             remove_item(item)
# #
# #     # def remove_item(item: tuple[float, tuple[int, int]]):
# #
# #
# #
# #     # while open_set:
# #     #     curr = heapq.heappop(open_set)
# #     #     for nbr in neighbour_coords(curr):
# #     #         if nbr[0] != curr[0]
# #
# #     return q_map, open_set
#
#
# class PathFinder:
#     def __init__(self, map_path, map_image_path, start_coords, end_coords, targets):
#         self.map_array = np.array(Image.open(map_path))
#         self.map_image = cv2.imread(map_image_path)
#         self.start = start_coords
#         self.end = end_coords
#         self.targets = self.generate_targets(targets)
#         self.all_nodes = set([self.start, self.end] + [x for xs in self.targets for x in xs])
#         self.graph = self.generate_graph()
#
#     def generate_targets(self, targets):
#         offsets = [(-85, 0), (0, 85), (85, 0), (0, -85), (-60, 60), (60, 60), (60, -60), (-60, -60)]
#         all_tgts = []
#         for tgt in targets:
#             tgt_set = []
#             for offset in offsets:
#                 result = tuple(map(lambda x, y: x + y, tgt, offset))
#                 if self.map_array[result] != 0:
#                     tgt_set.append(result)
#             all_tgts.append(tgt_set)
#         return all_tgts
#
#     def show_map(self):
#         output_map = cv2.cvtColor(self.map_image.copy(), cv2.COLOR_BGR2RGB)
#         for poi in self.all_nodes:
#             output_map[poi] = 255
#
#         Image.fromarray(output_map).show()
#
#     def generate_graph(self):
#         graph = \
#             {node:
#                 {nbr:
#                     {
#                         'distance': np.inf,
#                         'energy': np.inf
#                     }
#                     for nbr in self.all_nodes if nbr != node}
#                 for node in self.all_nodes}
#         del graph[self.start][self.end]
#         del graph[self.end][self.start]
#         return graph
#
#     def find_paths(self, start, targets):
#         g_values = np.full_like(self.map_array, np.inf)
#         e_values = np.full_like(self.map_array, np.inf)
#
#
# pathfinder = PathFinder("map_energy_quantised.bmp", "map_all.png",
#                         START, END, [TGT_A, TGT_B, TGT_C])
#
# print(a_star(START, END, pathfinder.map_array))
# print(pathfinder.map_array[(23, 234)])
#
#
# # for key in pathfinder.graph.keys():
# #     print(neighbour_coords(key, pathfinder.map))
#
#
# # print(euclidian_distance(START, END))
#
# # print([key for key in pathfinder.graph[START].keys()])
# # pathfinder.show_map()
#
# # offsets = tuple(map(lambda x, y: x - y, START, END))
# # distance_pixels = np.sqrt((offsets[0]**2) + (offsets[1]**2))
# # print(distance_pixels)
# # np_img = np.array(Image.open("map_energy_quantised.bmp"))
# #
# # unique, counts = np.unique(np_img, return_counts=True)
# # print(dict(zip(unique, counts)))
# #
# # print(neighbour_coords((2226, 4399), np_img))
# #
# # np_img[START] = 255
# # np_img[TGT_A] = 255
# # np_img[TGT_B] = 255
# # np_img[TGT_C] = 255
# # np_img[END] = 255
# #
# # coords_dict = {}
#
# # for i in range(0, np_img.shape[0]):
# #     for j in range(0, np_img.shape[1]):
# #         coords_dict[(i, j)] = None
#
# # print(coords_dict)
#
# # TODO Use arrays for all factors, g cost, e cost, node from coords tuple for distance and energy,
