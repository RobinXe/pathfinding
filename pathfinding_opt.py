import math
import os
import io
import av
import pickle
import ffmpeg
import numba
import cv2
from PIL import Image
import numpy as np
import heapq
import itertools
import concurrent.futures
import time
from tqdm import tqdm
from multiprocessing import Array

START = (551, 486)
END = (1435, 2469)
TGT_A = (791, 1441)
TGT_B = (982, 1551)
TGT_C = (1410, 2181)
MAP_IMAGE = cv2.imread("map_all.png")

sqrt2 = np.sqrt(2)


@numba.njit(fastmath=True)
def neighbour_coords(coords, map_array):
    nbs = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (i == 0 and j == 0):
                nbr = (coords[0] + i, coords[1] + j)
                if 0 <= nbr[0] < map_array.shape[0] and 0 <= nbr[1] < map_array.shape[1]:
                    if map_array[nbr] != 0:
                        nbs.append(nbr)
    return nbs


@numba.njit(fastmath=True)
def neighbour_coords_generalised(coords, map_array, radius):
    nbs = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if not (i == 0 and j == 0):
                nbr = (coords[0] + i, coords[1] + j)
                step_cost = np.sqrt(i ** 2 + j ** 2)
                if 0 <= nbr[0] < map_array.shape[0] and 0 <= nbr[1] < map_array.shape[1]:
                    if map_array[nbr] != 0:
                        nbs.append((nbr, step_cost))
    return nbs


@numba.njit(fastmath=True)
def euclidian_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@numba.njit(fastmath=True)
def diagonal_distance(p1, p2):
    dx = abs(p1[1] - p2[1])
    dy = abs(p1[0] - p2[0])

    return (dx + dy) + (sqrt2 - 2) * min(dx, dy)


def dijkstra(map_array, start, targets):
    shape = map_array.shape
    g_values = np.full(shape, np.inf, dtype=np.float32)
    e_values = np.full(shape, np.inf, dtype=np.float32)
    prev = np.full(shape, None)
    closed_set = set([])
    explored_history = []
    open_set = PQ()
    g_values[start] = 0  # Initialise with start point coords and values
    open_set.add_item(start, 0)
    e_values[start] = 0
    timer = time.time()

    remaining_targets = [i for i in targets if i not in closed_set]

    while remaining_targets:
        if len(closed_set) % 100000 == 0:
            print(f'Explored {len(closed_set)} nodes')

        current = open_set.pop_item()
        closed_set.add(current)
        explored_history.append(current)
        current_neighbours = neighbour_coords_generalised(current, map_array, 3)

        for neighbour in current_neighbours:
            if neighbour[0] not in closed_set:

                provisional_neighbour_e = e_values[current] + (neighbour[1] * map_array[neighbour[0]])

                if provisional_neighbour_e < e_values[neighbour[0]]:
                    e_values[neighbour[0]] = provisional_neighbour_e
                    g_values[neighbour[0]] = g_values[current] + neighbour[1]
                    prev[neighbour[0]] = current
                    open_set.add_item(neighbour[0], provisional_neighbour_e)

        remaining_targets = [i for i in targets if i not in closed_set]

    results = []
    for target in targets:
        path = []
        current = target

        while current != start:
            path.insert(0, current)
            current = prev[current]
        path.insert(0, start)

        results.append((start, target, e_values[target], g_values[target], np.array(path, dtype=np.uint16)))

    print(f'All targets found in {time.time() - timer} s.')

    return results, np.array(explored_history, dtype=np.uint16)


def a_star(start, end, map_array):
    heuristic = euclidian_distance
    shape = map_array.shape
    g_values = np.full(shape, np.inf, dtype=np.float32)  # Array to store values for g(n)
    # de_values = np.zeros(shape, dtype=np.float32)
    prev = np.full(shape, None)  # Array to store parent node
    closed_set = set([])  # List of explored nodes
    explored_history = []
    open_set = PQ()
    g_values[start] = 0  # Initialise with start point coords and values
    open_set.add_item(start, heuristic(start, end))

    while end not in closed_set:
        current = open_set.pop_item()
        closed_set.add(current)
        explored_history.append(current)
        current_neighbours = neighbour_coords_generalised(current, map_array, 5)

        if current == end:
            break

        for neighbour in current_neighbours:
            if neighbour[0] not in closed_set:
                neighbour_h = heuristic(neighbour[0], end)

                provisional_neighbour_g = g_values[current] + neighbour[1]

                if provisional_neighbour_g < g_values[neighbour[0]]:
                    g_values[neighbour[0]] = provisional_neighbour_g
                    prev[neighbour[0]] = current
                    open_set.add_item(neighbour[0], provisional_neighbour_g + neighbour_h)
                    # de_values[neighbour[0]] = de_values[current] + (neighbour[1] * map_array[neighbour[0]])

    path = []
    current = end

    while current != start:
        path.insert(0, current)
        current = prev[current]
    path.insert(0, start)

    return g_values[end], [np.array(path, dtype=np.uint16), ], np.array(explored_history, dtype=np.uint16)


def route_image(map_image, route):
    for px in route:
        map_image[px] = (255, 0, 255)

    output_map = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
    Image.fromarray(output_map).show()


def show_routes(image, routes):
    for rte in routes:
        for idx, _ in enumerate(rte):
            if (rte[idx] != rte[-1])[0]:
                pt1 = tuple([int(i) for i in reversed(rte[idx])])
                pt2 = tuple([int(i) for i in reversed(rte[idx + 1])])
                try:
                    cv2.line(image, pt1, pt2, (130, 0, 130), 3)
                except IndexError:
                    pass
    # image = image[:2160, :3840]

    return (image[::2, ::2]).astype(np.uint8)


def resize_frame(frame, ratio=0.436363637):
    return cv2.resize(frame, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)


def frame_from_array(frame):
    return av.VideoFrame.from_ndarray(frame, format="bgr24")


@numba.njit(parallel=True, fastmath=True)
def render_frame_opt(map_img, history, frame_no, slice_size):
    for i in numba.prange(slice_size):
        idx = i + (slice_size * frame_no)
        if idx <= len(history):
            coords = (history[i + (slice_size * frame_no)][0], history[i + (slice_size * frame_no)][1])
            # print(coords)
            map_img[coords] = (((map_img[coords] * 0.8) + np.array((51, 0, 51), dtype=np.uint8)).astype(np.uint8))
    return map_img


def render_video_opt(map_img, history, routes, vid_length, vid_fps, output_path):
    vid_no_frames = vid_length * vid_fps
    slice_size = int(math.ceil(len(history) / vid_no_frames))
    map_img = (map_img.copy()).astype(np.uint8)

    memory_file = io.BytesIO()
    output = av.open(memory_file, 'w', format='mp4')
    stream = output.add_stream('h264', str(vid_fps))
    stream.width = 1920
    stream.height = 1080
    stream.options = {'crf': '20'}

    for _ in range(vid_fps * 2):
        video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
        packet = stream.encode(video_frame)
        output.mux(packet)

    timer_render = time.time()
    print('Rendering frames...')
    time.sleep(0.01)
    for i in tqdm(range(vid_no_frames)):
        map_img = render_frame_opt(map_img, history, i, slice_size)
        video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
        packet = stream.encode(video_frame)
        output.mux(packet)
    print(f'{vid_no_frames} rendered in {time.time() - timer_render:2f} seconds.')

    map_img = show_routes(map_img, routes)

    for _ in range(vid_fps * 5):
        video_frame = av.VideoFrame.from_ndarray(map_img, format="bgr24")
        packet = stream.encode(video_frame)
        output.mux(packet)

    timer_save_video = time.time()
    print('Saving video...')
    packet = stream.encode(None)
    output.mux(packet)
    output.close()

    with open(output_path, 'wb') as f:
        f.write(memory_file.getbuffer())
    print(f'Video saved in {time.time() - timer_save_video:2f} seconds.')


@numba.njit(cache=False, parallel=True)
def render_frame_conc(data, frame_no, slice_size, final_frame=False):  # map_image: np.array([[[]]]),
    # global map_image
    # global data
    exp_history = data
    frame = MAP_IMAGE.copy()

    if frame_no == 0:
        return (frame[:2160:2, :3840:2]).astype(np.uint8)
    old_points = exp_history[:slice_size * frame_no - 1]
    new_points = exp_history[slice_size * frame_no - 1:slice_size * frame_no]

    if len(old_points) > 0:
        for pt in numba.prange(len(old_points)):
            frame[(old_points[pt][0], old_points[pt][1])] = (((frame[(old_points[pt][0], old_points[pt][1])] * 0.8)
                                                              + np.array((51, 0, 51), dtype=np.uint8))
                                                             .astype(np.uint8))

    if len(new_points) > 0:
        for pt in numba.prange(len(new_points)):
            frame[(new_points[pt][0], new_points[pt][1])] = np.array((255, 0, 255), dtype=np.uint8)

    if final_frame:
        return (frame[:2160, :3840]).astype(np.uint8)
    else:
        return (frame[:2160:2, :3840:2]).astype(np.uint8)


def build_frames(slice_size, frame_no):
    global explored_history
    history_list = []
    live_list = []
    final_frame_list = []
    history = np.array(explored_history[:slice_size * frame_no])
    if len(history) > 0:
        history_list.append(history)
    else:
        history_list.append(np.array([[9999999, 9999999], [9999999, 9999999]]))
    live_list.append(np.array(explored_history[slice_size * frame_no:slice_size * (frame_no + 1)]))
    final_frame_list.append(False)

    return [history_list, live_list, final_frame_list]


def init_frame_worker(map_arg, data_arg):
    global map_image_global
    map_image_global = map_arg
    global data
    data = data_arg
    # render_frame_conc.recompile()


def render_video_conc(show_route, exp_history, vid_length, output_path):  # show_route: list of lists
    vid_no_frames = vid_length * 60
    slice_size = len(exp_history) // vid_no_frames
    leftover_points = len(exp_history) % vid_no_frames
    history_list = []
    live_list = []
    final_frame_list = []
    timer = time.time()
    exp_history = np.array(exp_history)

    # for _ in range(120):
    #     history_list.append(np.array([[9999999, 9999999], [9999999, 9999999]]))
    #     live_list.append(np.array([[9999999, 9999999], [9999999, 9999999]]))
    #     final_frame_list.append(False)

    # for i in range(vid_no_frames - 1):
    #     history = np.array(explored_history[:slice_size * i])
    #     if len(history) > 0:
    #         history_list.append(history)
    #     else:
    #         history_list.append(np.array([[9999999, 9999999], [9999999, 9999999]]))
    #     live_list.append(np.array(explored_history[slice_size * i:slice_size * (i + 1)]))
    #     final_frame_list.append(False)

    frame_counter = [i for i in range(vid_no_frames + 3)]
    # slice_size_list = [slice_size, ] * (vid_no_frames + 2)
    final_frame_list = [False, ] * (vid_no_frames + 1) + [True]

    # with concurrent.futures.ProcessPoolExecutor(12, initializer=init_frame_worker, initargs=(MAP_IMAGE, exp_history)) as executor:
    #     results = [i for i in executor.map(build_frames, slice_size_list, frame_counter)]
    #
    # # history_list = history_list + [i[0] for i in results]
    # # live_list = live_list + [i[1] for i in results]
    # # final_frame_list = final_frame_list + [i[2] for i in results]
    # #
    # # if leftover_points > 0:
    # #     history_list.append(np.array(exp_history[:leftover_points]))
    # #     live_list.append(np.array(exp_history[-leftover_points:]))
    # #     final_frame_list.append(False)
    #
    # history_list.append(np.array(exp_history))
    # live_list.append(np.array([[9999999, 9999999], [9999999, 9999999]]))
    # final_frame_list.append(True)
    #
    # time_split = time.time() - timer
    # print(f'Frame pool built in {time_split} s.')
    # time_split = time.time()
    print('Rendering frames:')
    with concurrent.futures.ProcessPoolExecutor(12, initializer=init_frame_worker,
                                                initargs=(MAP_IMAGE, exp_history)) as executor:
        frames = list(tqdm(executor.map(render_frame_conc, itertools.repeat(exp_history), frame_counter,
                                        itertools.repeat(slice_size), final_frame_list), total=vid_no_frames + 2))

    # frames = [i for i in frames]
    final_frame = show_routes(frames[-1], show_route)
    del frames[-1]

    for i in range(300):
        frames.append(final_frame)

    for i in range(119):
        frames.insert(0, frames[0])

    time_split = time.time() - timer
    print(f'Frames rendered in {time_split} s.')
    time_split = time.time()

    print('Rendering video:')

    memory_file = io.BytesIO()

    output = av.open(memory_file, 'w', format='mp4')
    stream = output.add_stream('h264', '60')
    stream.width = 1920
    stream.height = 1080
    stream.options = {'crf': '20'}

    for frame in tqdm(frames):
        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        packet = stream.encode(video_frame)
        output.mux(packet)

    packet = stream.encode(None)
    output.mux(packet)
    output.close()

    with open(output_path, 'wb') as f:
        f.write(memory_file.getbuffer())

    time_split = time.time() - time_split
    print(f'Video saved in {time_split} s.')


def render_video(map_image, route, explored_history, vid_length):
    shape = map_image.shape
    d_shape = map_image.shape[:2]
    colour_full = np.full(shape, (255, 0, 255), dtype=np.uint8)
    old_alpha = np.zeros(d_shape, dtype=np.single)
    front_alpha = np.zeros(d_shape, dtype=np.single)

    vid_no_frames = vid_length * 60
    leftover_points = len(explored_history) % vid_no_frames
    slice_size = int((len(explored_history) - leftover_points) / vid_no_frames)
    remainder_history = explored_history

    pre_frame = cv2.resize(map_image, None, fx=0.436363637, fy=0.436363637, interpolation=cv2.INTER_CUBIC)
    video = cv2.VideoWriter('{}_{}.avi'.format(route[0], route[-1]), cv2.VideoWriter_fourcc(*'MJPG'),
                            60, (pre_frame.shape[1], pre_frame.shape[0]))

    for _ in range(120):
        video.write(pre_frame.astype(np.uint8))

    control = True
    while control:

        if len(remainder_history) > slice_size:
            current_slice = remainder_history[:slice_size]
            remainder_history = remainder_history[slice_size:]
        elif remainder_history:
            current_slice = remainder_history
            remainder_history = []
        else:
            current_slice = []
            control = False

        for coord in current_slice:
            front_alpha[coord] = 1

        current_alpha = old_alpha + front_alpha
        current_mask = np.dstack((current_alpha, current_alpha, current_alpha))

        current_frame = (map_image * (1 - current_mask) + colour_full * current_mask).astype(np.uint8)
        current_frame_resized = cv2.resize(current_frame,
                                           None, fx=0.436363637, fy=0.436363637, interpolation=cv2.INTER_CUBIC)
        video.write(current_frame_resized)
        # print('{} frames of {} processed'.format(len(explored_history)-len(remainder_history), len(explored_history)))

        front_alpha = front_alpha * 0.3
        old_alpha = old_alpha + front_alpha
        front_alpha = np.zeros(d_shape, dtype=np.single)

    for _ in range(30):
        video.write(current_frame_resized)

    for i in route:
        cv2.circle(current_frame, (i[1], i[0]), 3, (130, 0, 130), 2)
        current_frame_resized = cv2.resize(current_frame, None, fx=0.436363637, fy=0.436363637,
                                           interpolation=cv2.INTER_CUBIC)

    for _ in range(120):
        video.write(current_frame_resized)

    video.release()


class PQ:
    def __init__(self):
        self.pq = []
        self.map_dict = {}
        self.REMOVED = '<entry removed>'
        self.counter = itertools.count()

    def add_item(self, item, priority):
        if item in self.map_dict:
            self.remove_item(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.map_dict[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.map_dict.pop(item)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.map_dict[item]
                return item
        raise KeyError('pop from an empty priority queue')


class PathFinder:
    def __init__(self, map_path, map_image_path, start_coords, end_coords, targets):
        self.map_array = np.array(Image.open(map_path))[:2160, :3840]
        self.map_image = cv2.imread(map_image_path)[:2160, :3840]
        self.start = start_coords
        self.end = end_coords
        self.targets = self.generate_targets(targets)
        self.all_target_sets = [[self.start, self.end], ] + self.targets
        self.all_nodes = set([self.start, self.end] + [x for xs in self.targets for x in xs])
        self.graph = self.generate_graph()
        self.pixel_nm = 2672.246 / euclidian_distance(self.start, self.end)
        self.energy_divisor = 150 / (4949 / euclidian_distance(self.start, self.end))

    def empty_dicts(self, field='distance'):
        count = 0
        for k in self.graph.keys():
            for j in self.graph[k].keys():
                if self.graph[k][j][field] == np.inf:
                    count += 1
        return count

    def generate_targets(self, targets):
        offsets = [(-85, 0), (0, 85), (85, 0), (0, -85), (-60, 60), (60, 60), (60, -60), (-60, -60)]
        all_tgts = []
        for tgt in targets:
            tgt_set = []
            for offset in offsets:
                result = tuple(map(lambda x, y: x + y, tgt, offset))
                if self.map_array[result] != 0:
                    tgt_set.append(result)
            all_tgts.append(tgt_set)
        return all_tgts

    def show_map(self):
        output_map = cv2.cvtColor(self.map_image.copy(), cv2.COLOR_BGR2RGB)
        for poi in self.all_nodes:
            output_map[poi] = 255

        Image.fromarray(output_map).show()

    def generate_graph(self):
        graph = {}

        for n in range(len(self.all_target_sets)):
            for i in self.all_target_sets[n]:
                graph[i] = {}
                for j in self.all_nodes:
                    if j not in self.all_target_sets[n]:
                        graph[i][j] = {
                            'distance': np.inf,
                            'd_path': None,
                            'energy': np.inf,
                            'e_path': None
                        }

        return graph

    def find_distances(self):
        pairs = set([])
        for start in self.graph.keys():
            for end in self.graph[start].keys():
                if (end, start) not in pairs:
                    pairs.add((start, end))

        starts, ends = list(zip(*pairs))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(a_star, starts, ends, itertools.repeat(self.map_array)):
                start = result[1][0]
                end = result[1][-1]
                self.graph[start][end]['distance'] = result[0] * self.pixel_nm
                self.graph[start][end]['d_path'] = result[1]
                self.graph[end][start]['distance'] = self.graph[start][end]['distance']
                self.graph[end][start]['d_path'] = list(reversed(self.graph[start][end]['d_path']))

    def generate_target_permutations(self, target_set=None):

        if not target_set:
            target_set = self.targets.copy()

        if len(target_set) == 1:
            return [[t, ] for t in target_set[0]]
        else:
            res = []
            for s in target_set:
                remainder_set = target_set.copy()
                remainder_set.remove(s)

                for t in s:
                    for next_layer in self.generate_target_permutations(remainder_set):
                        res.append([t, ] + next_layer)

            return res

    def find_mst(self, parameter='distance'):

        if self.empty_dicts() > 0:
            self.find_distances()

        lowest_score = np.inf
        lowest_scoring_path = None
        for i in self.generate_target_permutations():
            i.insert(0, self.start)
            i.append(self.end)

            score = 0
            for idx, _ in enumerate(i):
                if idx <= len(i) - 2:
                    score += self.graph[i[idx]][i[idx + 1]][parameter]
            if score < lowest_score:
                lowest_score = score
                lowest_scoring_path = i

        return lowest_score, lowest_scoring_path

    def display_route(self, path, parameter='d_path'):

        frame = self.map_image.copy()

        route = []
        for idx in range(len(path) - 1):
            route.append(self.graph[path[idx]][path[idx + 1]][parameter])

        for bound in route:
            for idx in range(len(bound) - 1):
                cv2.line(frame, tuple(reversed(bound[idx])), tuple(reversed(bound[idx + 1])), (130, 0, 130), 3)

        output_map = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(output_map).show()


if __name__ == '__main__':

    # pathfinder = PathFinder("map_energy_quantised.bmp", "map_all.png",
    #                         START, END, [TGT_A, TGT_B, TGT_C])
    #
    # print(pathfinder.pixel_nm)
    # print(pathfinder.energy_divisor)
    #
    # length, steps = pathfinder.find_mst()
    #
    # # print(f'Path Length is {length} nm')
    # # pathfinder.display_route(steps)
    #
    # with open('pathfinder.pkl.bak', 'wb') as file:
    #     pickle.dump(pathfinder, file)

    try:
        with open('pathfinder.pkl', 'rb') as file:
            pathfinder = pickle.load(file)
    except FileNotFoundError:
        pathfinder = PathFinder("map_energy_quantised.bmp", "map_all.png",
                                START, END, [TGT_A, TGT_B, TGT_C])

    # timer = time.time()
    # X = Array('i', pathfinder.map_image.shape[0] * pathfinder.map_image.shape[1] * pathfinder.map_image.shape[2])
    #
    # X_np = np.frombuffer(X.get_obj(), dtype=np.int32).reshape(pathfinder.map_image.shape).astype(np.uint8)
    #
    # np.copyto(X_np, pathfinder.map_image)
    # timer = time.time() - timer
    # print(f'Completed the memory operations in {timer:.2f} seconds.')
    #
    # cv2.imshow('MAP', X_np[::2, ::2])
    # cv2.waitKey(0)

    # print(pathfinder.pixel_nm)

    t_list = [i for i in pathfinder.graph[START].keys()]

    res, exp = dijkstra(pathfinder.map_array, START, t_list)

    # res, exp = dijkstra(pathfinder.map_array, START, [(750, 750)])

    render_video_opt(pathfinder.map_image, exp, [r[-1] for r in res], 180, 60, 'output2.mp4')

    # render_video_conc(pathfinder.map_image, [r[-1] for r in res], exp, 60)

    # render_frame.recompile()
    # render_video_conc([r[-1] for r in res], exp, 10, 'output1.mp4')

    # print(START, [k for k in pathfinder.graph[START].keys()][0])
    # print(pathfinder.graph[START][[k for k in pathfinder.graph[START].keys()][0]]['d_path'])
    #
    # print(pathfinder.targets)

    # pathfinder.find_distances()
    #
    # for start in pathfinder.graph.keys():
    #     for end in pathfinder.graph[start].keys():
    #         if not pathfinder.graph[start][end]['distance']:
    #             print(start, end)

    # for i in neighbour_coords_generalised((3, 3), pathfinder.map_array, 3):
    #     print(i[1])
    #
    # print(len(neighbour_coords_generalised((3, 3), pathfinder.map_array, 3)))

    # x = [1, 2, 3, 4, 5]
    # print(x[:len(x)])

    # x = np.full((2, 2, 3), 100, dtype=np.single)
    # y = np.full((2, 2), 0, dtype=np.single)
    #
    # print(x*y)

    # distance, route, explored = a_star(START, END, pathfinder.map_array)
    #
    # print(distance)
    # print(len(route))
    # print(len(explored))
    #
    # render_video_conc(pathfinder.map_image, route, explored, 30)

    # print(distance)
    # print(4949/euclidian_distance(START, END))
    # print(distance * (4949/euclidian_distance(START, END)))
    #
    # print(len(explored))

    # route_image(pathfinder.map_image, path)
    # print(len(pathfinder.graph[list(pathfinder.graph.keys())[2]]))

    # print(pathfinder.graph)

    # print(len(list(chain.from_iterable([list(pathfinder.graph[i].keys()) for i in pathfinder.graph.keys()]))))
    # print(list(chain.from_iterable([list(pathfinder.graph[i].keys()) for i in pathfinder.graph.keys()])))
    # print(((24*18))+48)
    # print(len(pathfinder.graph.keys()))
    # for i in pathfinder.graph.keys():
    #     print(len(pathfinder.graph[i]))
    # print(pathfinder.graph[START][list(pathfinder.all_nodes)[3]])

    # for key in pathfinder.graph.keys():
    #     print(neighbour_coords(key, pathfinder.map))

    # print(euclidian_distance(START, END))

    # print([key for key in pathfinder.graph[START].keys()])
    # pathfinder.show_map()

    # offsets = tuple(map(lambda x, y: x - y, START, END))
    # distance_pixels = np.sqrt((offsets[0]**2) + (offsets[1]**2))
    # print(distance_pixels)
    # np_img = np.array(Image.open("map_energy_quantised.bmp"))
    #
    # unique, counts = np.unique(np_img, return_counts=True)
    # print(dict(zip(unique, counts)))
    #
    # print(neighbour_coords((2226, 4399), np_img))
    #
    # np_img[START] = 255
    # np_img[TGT_A] = 255
    # np_img[TGT_B] = 255
    # np_img[TGT_C] = 255
    # np_img[END] = 255
    #
    # coords_dict = {}

    # for i in range(0, np_img.shape[0]):
    #     for j in range(0, np_img.shape[1]):
    #         coords_dict[(i, j)] = None

    # print(coords_dict)

    # TODO Use arrays for all factors, g cost, e cost, node from coords tuple for distance and energy,

    # https://santhalakshminarayana.github.io/blog/super-fast-python-multi-processing
    # https://santhalakshminarayana.github.io/blog/super-fast-python-numba
    # https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
    # https://stackoverflow.com/questions/66937630/python-multiprocessing-with-shared-rawarray
    # https://stackoverflow.com/questions/66378848/passing-shared-memory-variables-in-python-multiprocessing/66380200#66380200
    # https://numpy.org/doc/stable/reference/generated/numpy.copy.html
