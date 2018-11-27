import csv, random, math, queue
from Node import Node
from random import choice
from Draw import draw_decision_tree, draw_plot_for_labels
import numpy as np


class Decisive_Tree:
    def __init__(self, file):
        self.file = file
        self.columns, self.used_in_tree = [], []
        self.values, self.train, self.valid, self.test = np.array([]), np.array([]), np.array([]), np.array([])
        self.features, self.root, self.nodesCnt, self.threshold = 0, 0, 0, 8
        self.root = None

    def create_dataset(self, train_ratio=5, valid_ratio=1, test_ratio=1):
        with open(self.file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            splitted_lines = [x[0].split(';') for x in csv_reader]
            self.columns = [str(el.replace('"', '')) for el in splitted_lines[0]]
            self.features = len(splitted_lines[0])
            self.used_in_tree = [False] * self.features
            for line in splitted_lines[1:4000]:
                line = [float(el) for el in line]
                self.values = np.concatenate(self.values, np.asarray(line))
        self.extract_datasets(train_ratio, valid_ratio, test_ratio)
        
    def extract_datasets(self, train_ratio, valid_ratio, test_ratio):
        n = self.values.size
        all_ratio = train_ratio + test_ratio + valid_ratio
        np.random.shuffle(self.values)
        self.train = np.asarray(self.values[:int((train_ratio*n)/all_ratio)])
        self.valid = np.asarray(self.values[int((train_ratio*n)/all_ratio):int(((train_ratio+valid_ratio)*n)/all_ratio)])
        self.test = np.asarray(self.values[int(((train_ratio+valid_ratio)*n)/all_ratio):])

    def get_labels(self, data):
        labels = data[:, -1]
        return (labels, np.unique(labels))

    def get_distro(self, data):
        labels = data[:, -1] # czy to dziala?
        u_labels, counts = np.unique(labels, return_counts=True)
        return counts/labels.size # broadcast?

    def compute_entropy(self, data):
        distro = self.get_distro(data)
        return 0 if np.count_nonzero(distro) else sum(-1*pi*math.log(pi, 2) for pi in distro)

    def information_gain(self, val, entropy, data):
        smaller = np.where(data <= val)
        greater = np.where(data > val)
        prob_smaller = smaller.size / (smaller.size + greater.size)
        prob_greater = greater.size / (smaller.size + greater.size)
        s_entropy = self.compute_entropy(smaller)
        g_entropy = self.compute_entropy(greater)
        return (entropy - (prob_smaller*s_entropy + prob_greater*g_entropy), smaller, greater)

    def feature_choice(self, data, used_already):
        maxIG = (-1, 0, 0, [], [])
        entropy = self.compute_entropy(data)
        for ix,iy in np.ndindex(data.shape): # byc moze po prostu podwojna petla bedzie miala lepszy performance
            if not used_already[iy]:
                inf_gain = self.information_gain(data[ix, iy], entropy, data)
                if inf_gain[0] < 0:
                        print("Oho... mam buga...", inf_gain[0])
                if maxIG[0] < inf_gain[0]:
                    maxIG = (inf_gain[0], ix, iy, inf_gain[1], inf_gain[2])
        used_already[maxIG[2]] = True
        return maxIG[1:]

    def set_root_node(self, fea_id, split_id):
        self.used_in_tree[fea_id] = True
        self.root = Node(self.used_in_tree)
        self.root.set_all(self.train[split_id][fea_id], fea_id, self.nodesCnt, self.train)

    def setup_node(self, data, node, fea_id, split_id):
        node.used_feature(fea_id)
        node.set_all(data[split_id][fea_id], fea_id, self.nodesCnt, data)
        self.nodesCnt += 1

    def create_tree(self):
        fea_id, split_id, s, g = self.feature_choice(self.train, self.used_in_tree)
        self.set_root_node(fea_id, split_id)
        que = queue.Queue()
        que.put((self.root, 'l', s, 1))
        que.put((self.root, 'r', g, 1))
        while not que.empty():
            node, side, data, depth = que.get()
            if depth > self.threshold or not len(data):
                continue
            node.create_left(node.get_used()) if side == 'l' else node.create_right(node.get_used())
            node = node.left if side == "left" else node.right
            fea_id, split_id, s, g = self.feature_choice(data, node.get_used())
            self.setup_node(data, node, fea_id, split_id)
            que.put((node, 'l', s, depth+1))
            que.put((node, 'r', g, depth+1))

    def calculate_accuracy(self, result, predicted_result):
        return np.where(result == predicted_result).size/result.size
    
    def set_node(self, t, fea_id):
        if t.left is None or t.right is None:
            return t.left if t.right is None else t.right
        else:
            return t.right if fea_id > t.threshold else t.left

    def evaluate_tree(self):
        predicted_result, result = [], []
        for sample in self.test:
            t = self.root
            while True:
                if t.left is None and t.right is None:
                    u_labels, cnt_labels = np.unique(t.labels, return_counts=True)
                    label = max(np.asarray(u_labels, cnt_labels), key=lambda x: x[1])
                    result.append(int(sample[-1]))
                    predicted_result.append(int(label[0]))
                    break
                else:
                    t = self.set_node(t, sample[t.feature_id])
        result, predicted_result = np.asarray(result), np.asarray(predicted_result)
        return ("Accuracy obtained on test data is %f" % self.calculate_accuracy(result, predicted_result), result, predicted_result)

    def prune_tree(self):
        pass