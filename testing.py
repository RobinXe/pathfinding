import numba
import numpy as np
import heapq
from queue import PriorityQueue
import itertools


@numba.njit
def test_sort():
    l = [(199, (465, 789)), (15, (2734, 2598)), (17, (236, 982)), (15, (234, 6545))]
    heapq.heapify(l)
    counter = itertools.count()
    return heapq.heappop(l)


# class PQ:

    # def __init__(self):
    #     self.pq = []
    #     self.map_dict = {}
    #     self.REMOVED = '<entry removed>'
    #
    # def add_item(self, item, priority):
    #     if item in self.map_dict:
    #         self.remove_item(item)
    #     entry = [priority, item]
    #     self.map_dict[item] = entry
    #     heapq.heappush(self.pq, entry)
    #
    # def remove_item(self, item):
    #     entry = self.map_dict.pop(item)
    #     entry[-1] = self.REMOVED
    #
    # def pop_item(self):
    #     while self.pq:
    #         priority, item = heapq.heappop(self.pq)
    #         if item is not self.REMOVED:
    #             del self.map_dict[item]
    #             return item
    #     raise KeyError('pop from an empty priority queue')


def mathsy():
    return 24*16*8

def permutationsy():
    A_set = ["A" + str(i) for i in range(1, 9)]
    B_set = ["B" + str(i) for i in range(1, 9)]
    C_set = ["C" + str(i) for i in range(1, 9)]

    all_set = [A_set, B_set, C_set]
    perms=[]

    for i in range(0, len(all_set)):
        remainder_1 = all_set.copy()
        set_1 = remainder_1.pop(i)
        for x in set_1:

            for j in range(0, len(remainder_1)):
                remainder2 = remainder_1.copy()
                set_2 = remainder2.pop(j)
                for y in set_2:

                    for k in range(0, len(remainder2)):
                        remainder3 = remainder2.copy()
                        set_3 = remainder3.pop(k)
                        for z in set_3:
                            perms.append([x, y, z])

    return perms

print(mathsy())
print(len(permutationsy()))

# print(test_sort())


# q = PQ()
# q.add_item((24, 24), 24)
# q.add_item((30, 30), 30)
# q.add_item((50, 50), 50)
# q.add_item((24, 24), 60)
# print(q.pop_item())