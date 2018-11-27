from copy import deepcopy
import numpy as np

class Node:
    def __init__(self, used_already):
        # brakuje tutaj labela, jesli node jest lisciem to ktorys numer od 1 do 10, jesli nie to -1
        # niekoniecznie musi byc jednoznacznosc, wiec fajnie zrobic losowanko, gdy mozliwosc jest wiecej
        # na przyklad gdy depth nie pozwala na wieksza dokladnosc
        self.threshold, self.feature_id, self.node_id = -1, -1, -1
        self.used_already = used_already
        self.labels = np.array([])
        self.left, self.right = None, None
    def __repr__(self):
        return 'Node for %d feature' % self.feature_id
    def set_all(self, threshold, feature_id, node_id, data):
        self.threshold = threshold
        self.feature_id = feature_id
        self.node_id = node_id
        self.labels = [data[i][-1] for i in range(len(data))]
    def is_set(self, id):
        return self.used_already[id]
    def used_feature(self, id):
        self.used_already[id] = True
    def create_left(self, used_already):
        self.left = Node(deepcopy(used_already))
    def create_right(self, used_already):
        self.right = Node(deepcopy(used_already))
    def get_used(self):
        return self.used_already