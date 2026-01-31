import copy
import heapq
import io
import logging
import math
from typing import Dict, List, Optional
import zlib
import torch

from strategies.client.backdoor_heuristics.heuristic import Heuristic, HeuristicOutput


class PriorityQueueItem:
    def __init__(self, score, obj, obj_hash):
        self.score = score
        self.obj = obj
        self.obj_hash = obj_hash

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return f"PriorityQueueItem(name={self.name}, score={self.score})"


class PriorityQueue:
    heap: List[PriorityQueueItem]

    def __init__(self, size):
        self.size = size
        self.heap = []
        self.hashes = set()

    def clear(self):
        objs = [item.obj for item in self.heap]
        self.heap.clear()
        self.hashes.clear()
        return objs

    def insert(self, score, obj, obj_hash):
        if math.isnan(score):
            return False, None

        if obj_hash in self.hashes:
            return False, None

        if not self.is_full():
            heapq.heappush(self.heap, PriorityQueueItem(score, obj, obj_hash))
            self.hashes.add(obj_hash)
            return True, None
        else:
            if score > self.heap[0].score:
                item = heapq.heapreplace(
                    self.heap, PriorityQueueItem(score, obj, obj_hash)
                )
                self.hashes.add(obj_hash)
                self.hashes.remove(item.obj_hash)
                return True, item.obj

        return False, None

    def is_empty(self):
        return len(self.heap) == 0

    def is_full(self):
        return len(self.heap) == self.size
