from copy import deepcopy
import numpy as np

class Node:
    def __init__(self):
        self.threshold, self.feature_id, self.node_id = -1, -1, -1
        self.labels = np.array([])
        self.left, self.right = None, None
        self.prune_flag = True
    def __repr__(self):
        return 'Node %d' % self.node_id
        #return 'Node for %d feature' % self.feature_id
    def set_all(self, threshold, feature_id, node_id, data):
        self.threshold, self.feature_id, self.node_id = threshold, feature_id, node_id
        self.labels = [data[i][-1] for i in range(len(data))]
    def create_left(self):
        self.left = Node()
    def create_right(self):
        self.right = Node()