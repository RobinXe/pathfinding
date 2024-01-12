import math
import io
import av
import pickle
import numba
import cv2
from PIL import Image
import numpy as np
import heapq
import itertools
import concurrent.futures
import time
from tqdm import tqdm

# Constants
START = (551, 486)
END = (1435, 2469)
TGT_A = (791, 1441)
TGT_B = (982, 1551)
TGT_C = (1410, 2181)
SQRT2 = np.sqrt(2)  # Used a lot; precompute to reduce load


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
    return np.sqrt(np.add(np.square(np.subtract(p1[0], p2[0])), np.square(np.subtract(p1[1], p2[1]))))


@numba.njit(fastmath=True)
def find_coord_in_array(array, item):
    for i in range(len(array)):
        if array[i][0] == item[0] and array[i][1] == item[1]:
            return i
    return -1


def dijkstra(map_array, start, targets, radius=3):
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
        # if len(closed_set) % 100000 == 0:
        #     print(f'Explored {len(closed_set)} nodes')

        current = open_set.pop_item()
        closed_set.add(current)
        explored_history.append(current)
        current_neighbours = neighbour_coords_generalised(current, map_array, radius)

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

    # print(f'All targets found in {time.time() - timer} s.')

    return results, np.array(explored_history, dtype=np.uint16)


def a_star(start, end, map_array, radius):
    heuristic = euclidian_distance
    shape = map_array.shape
    g_values = np.full(shape, np.inf, dtype=np.float32)  # Array to store values for g(n)
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
        current_neighbours = neighbour_coords_generalised(current, map_array, radius)

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


def show_routes(image, routes):
    for rte in routes:
        for idx, _ in enumerate(rte):
            if tuple(rte[idx]) != tuple(rte[-1]):
                pt1 = tuple([int(i) for i in reversed(rte[idx])])
                pt2 = tuple([int(i) for i in reversed(rte[idx + 1])])
                try:
                    cv2.line(image, pt1, pt2, (130, 0, 130), 3)
                except IndexError:
                    pass

    return image.astype(np.uint8)


@numba.njit(parallel=True, fastmath=True)
def render_frame_opt(map_img, history, frame_no, slice_size):
    for i in numba.prange(slice_size):
        idx = i + (slice_size * frame_no)
        if idx < len(history):
            coords = (history[i + (slice_size * frame_no)][0], history[i + (slice_size * frame_no)][1])
            if map_img[coords][0] != 130 or map_img[coords][1] != 0 or map_img[coords][2] != 130:
                map_img[coords] = (((map_img[coords] * 0.8) + np.array((51, 0, 51),
                                                                       dtype=np.uint8)).astype(np.uint8))
    return map_img


def render_video_opt(map_img, history, routes, vid_length, vid_fps, output_path, post_roll=5, mem_file=None):
    vid_no_frames = vid_length * vid_fps
    slice_size = int(math.ceil(len(history) / vid_no_frames))
    map_img = (map_img.copy()).astype(np.uint8)

    if mem_file is None:
        memory_file = io.BytesIO()
        output = av.open(memory_file, 'w', format='mp4')
        stream = output.add_stream('hevc', str(vid_fps))
        stream.width = 1920
        stream.height = 1080
        stream.options = {'crf': '20'}
    else:
        memory_file = mem_file
        memory_file.seek(0)
        output = av.open(memory_file, 'w', format='mp4')
        if len(output.streams) == 0:
            stream = output.add_stream('hevc', str(vid_fps))
            stream.width = 1920
            stream.height = 1080
            stream.options = {'crf': '20'}
        else:
            print('Found a stream')
            stream = output.streams.video[0]

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
    print(f'{vid_no_frames} frames rendered in {time.time() - timer_render:2f} seconds.')

    map_img = show_routes(map_img, routes)

    for _ in range((vid_fps * post_roll) + 1):
        video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
        packet = stream.encode(video_frame)
        output.mux(packet)

    timer_save_video = time.time()
    if mem_file is None:
        packet = stream.encode(None)
        output.mux(packet)
        output.close()

    if output_path:
        print('Saving video...')
        with open(output_path, 'wb') as f:
            f.write(memory_file.getbuffer())
        print(f'Video {output_path} saved in {time.time() - timer_save_video:.2f} seconds.')

    return map_img, memory_file  # .getbuffer()


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
        self.histories = {i: None for i in self.graph.keys()}

    def pop_and_pik(self):
        if self.empty_dicts('distance') > 0:
            self.find_distances()
        if self.empty_dicts('energy') > 0:
            self.find_energies()
        with open(f'{self.start}_{self.end}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def empty_dicts(self, field='distance'):
        count = 0
        for k in self.graph.keys():
            for j in self.graph[k].keys():
                if self.graph[k][j][field] == np.inf or self.graph[k][j][field] is None:
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
                            'd_history': None,
                            'energy': np.inf,
                            'avg_energy': np.inf,
                            'e_path': None,
                            'e_history': None
                        }

        return graph

    def find_energies(self):
        starts_targets = {i: set([j for j in self.graph[i].keys()]) for i in self.graph.keys()}

        for k in starts_targets.keys():
            for t in starts_targets[k]:
                if k in starts_targets[t]:
                    starts_targets[t].remove(k)

        starts_targets = {k: v for k, v in starts_targets.items() if len(v) != 0}
        starts = [i for i in starts_targets.keys()]
        targets = [starts_targets[k] for k in starts]

        print('Finding energies...')
        time.sleep(0.01)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            mp = itertools.repeat(self.map_array)
            rx = itertools.repeat(5)
            for results, history in list(tqdm(executor.map(dijkstra, mp, starts, targets, rx),
                                              total=len(starts))):
                history_loc = results[0][0]
                self.histories[history_loc] = history

                for result in results:
                    start = result[0]
                    end = result[1]
                    self.graph[start][end]['energy'] = result[2] / self.energy_divisor
                    self.graph[start][end]['avg_energy'] = result[2] / result[3]
                    self.graph[start][end]['e_path'] = result[4]
                    self.graph[start][end]['e_history'] = history_loc
                    self.graph[end][start]['energy'] = self.graph[start][end]['energy']
                    self.graph[end][start]['avg_energy'] = self.graph[start][end]['avg_energy']
                    self.graph[end][start]['e_path'] = list(reversed(self.graph[start][end]['e_path']))
                    self.graph[end][start]['e_history'] = history_loc

        self.histories = {k: v for k, v in self.histories.items() if v is not None}

    def find_distances(self):
        pairs = set([])
        for start in self.graph.keys():
            for end in self.graph[start].keys():
                if (end, start) not in pairs:
                    pairs.add((start, end))

        starts, ends = list(zip(*pairs))

        print('Finding distances...')
        time.sleep(0.01)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            mp = itertools.repeat(self.map_array)
            rx = itertools.repeat(5)
            for result in list(tqdm(executor.map(a_star, starts, ends, mp, rx),
                               total=len(starts))):
                start = tuple(result[1][0][0])
                end = tuple(result[1][0][-1])
                self.graph[start][end]['distance'] = result[0] * self.pixel_nm
                self.graph[start][end]['d_path'] = result[1]
                self.graph[start][end]['d_history'] = result[2]
                self.graph[end][start]['distance'] = self.graph[start][end]['distance']
                self.graph[end][start]['d_path'] = list(reversed(self.graph[start][end]['d_path']))
                self.graph[end][start]['d_history'] = self.graph[start][end]['d_history']

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

        if self.empty_dicts(parameter) > 0:
            if parameter == 'distance':
                self.find_distances()
            elif parameter == 'energy':
                self.find_energies()

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
            # print(self.graph[path[idx]][path[idx + 1]]['avg_energy'] * self.energy_divisor)

        for bound in route:
            for idx in range(len(bound) - 1):
                cv2.line(frame, tuple(reversed(bound[idx])), tuple(reversed(bound[idx + 1])), (130, 0, 130), 2)

        output_map = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(output_map).show()

    def video_route(self, route_type, vid_fps=30):
        score, route = self.find_mst(route_type)
        map_img = self.map_image.copy()

        memory_file = io.BytesIO()
        output = av.open(memory_file, 'w', format='mp4')
        # out_stream = output.streams.video[0]
        stream = output.add_stream('hevc', vid_fps)
        stream.width = 1920
        stream.height = 1080
        stream.options = {'crf': '20'}

        for _ in range(vid_fps * 2):
            video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
            packet = stream.encode(video_frame)
            output.mux(packet)

        if route_type == 'distance':
            tot_vid_no_frames = 60 * vid_fps
            tot_hist = sum([len(self.graph[route[i]][route[i + 1]]['d_history']) for i in range(len(route) - 1)])

            for idx, pt in enumerate(route):
                if pt != route[-1]:
                    hist = self.graph[pt][route[idx + 1]]['d_history']
                    rtes = self.graph[pt][route[idx + 1]]['d_path']
                    vid_no_frames = min(max(int(math.ceil(tot_vid_no_frames * (len(hist)/tot_hist))),
                                            (2 * vid_fps)),
                                        20 * vid_fps)
                    slice_size = int(math.ceil(len(hist) / vid_no_frames))

                    for i in tqdm(range(vid_no_frames)):
                        map_img = render_frame_opt(map_img, hist, i, slice_size)
                        video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
                        packet = stream.encode(video_frame)
                        output.mux(packet)

                    map_img = show_routes(map_img, rtes)

                    if route[idx + 1] == route[-1]:
                        p_r = 5
                    else:
                        p_r = 0

                    for _ in range((vid_fps * p_r) + 1):
                        video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
                        packet = stream.encode(video_frame)
                        output.mux(packet)

        elif route_type == 'energy':
            tot_vid_no_frames = 60 * 30
            tot_hist = 0

            for idx, pt in enumerate(route):
                if pt != route[-1]:
                    hist_loc = self.graph[pt][route[idx + 1]]['e_history']
                    hist = self.histories[hist_loc]
                    hist_stop = max(find_coord_in_array(hist, pt), find_coord_in_array(hist, route[idx + 1]))
                    tot_hist += len(hist[:hist_stop])

            for idx, pt in enumerate(route):
                if pt != route[-1]:
                    hist_loc = self.graph[pt][route[idx + 1]]['e_history']
                    hist = self.histories[hist_loc]
                    hist_stop = max(find_coord_in_array(hist, pt), find_coord_in_array(hist, route[idx + 1]))
                    rtes = self.graph[pt][route[idx + 1]]['e_path']
                    vid_no_frames = min(max(int(math.ceil(tot_vid_no_frames * (len(hist[:hist_stop]) / tot_hist))),
                                            (2 * vid_fps)),
                                        40 * vid_fps)
                    slice_size = int(math.ceil(len(hist[:hist_stop]) / vid_no_frames))

                    for i in tqdm(range(vid_no_frames)):
                        map_img = render_frame_opt(map_img, hist[:hist_stop], i, slice_size)
                        video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
                        packet = stream.encode(video_frame)
                        output.mux(packet)

                    map_img = show_routes(map_img, [rtes, ])

                    if route[idx + 1] == route[-1]:
                        p_r = 5
                    else:
                        p_r = 0

                    for _ in range((vid_fps * p_r) + 1):
                        video_frame = av.VideoFrame.from_ndarray(map_img[::2, ::2], format="bgr24")
                        packet = stream.encode(video_frame)
                        output.mux(packet)

        else:
            raise ValueError('Invalid route type. Must be "distance" or "energy".')

        packet = stream.encode(None)
        output.mux(packet)
        output.close()

        with open(f'{self.start}_{self.end}_{route_type}.mp4', 'wb') as f:
            f.write(memory_file.getbuffer())

        memory_file.close()


if __name__ == '__main__':

    try:
        with open(f'{START}_{END}.pkl', 'rb') as file:
            pathfinder = pickle.load(file)
    except FileNotFoundError:
        pathfinder = PathFinder("map_energy_quantised_fixed.bmp", "map_all.png",
                                START, END, [TGT_A, TGT_B, TGT_C])

        pathfinder.pop_and_pik()

    # pathfinder.video_route('energy')

# TODO Writeup

# https://santhalakshminarayana.github.io/blog/super-fast-python-multi-processing
# https://santhalakshminarayana.github.io/blog/super-fast-python-numba
# https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
# https://stackoverflow.com/questions/66937630/python-multiprocessing-with-shared-rawarray
# https://stackoverflow.com/questions/66378848/passing-shared-memory-variables-in-python-multiprocessing/66380200#66380200
# https://numpy.org/doc/stable/reference/generated/numpy.copy.html
