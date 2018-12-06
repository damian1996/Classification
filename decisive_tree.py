import csv, random, math as m, queue, numpy as np
from Node import Node

class Decisive_Tree:
    def __init__(self, file):
        self.file = file
        self.values, self.train, self.valid, self.test = np.array([]), np.array([]), np.array([]), np.array([])
        self.features, self.root, self.nodesCnt, self.depth_threshold = 0, 0, 0, 10
        self.to_preserve, self.to_remove = 0, 1
        self.root = None

    def create_dataset(self, train_ratio=5, valid_ratio=1, test_ratio=1):
        with open(self.file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            splitted_lines = [x[0].split(';') for x in csv_reader]
            self.features = len(splitted_lines[0]) - 1
            for line in splitted_lines[1:]:
                line = [float(el) for el in line]
                self.values = np.asarray(line) if self.values.size == 0 else np.vstack((self.values, np.asarray(line)))
        self.extract_datasets(train_ratio, valid_ratio, test_ratio)
        
    def extract_datasets(self, train_ratio, valid_ratio, test_ratio):
        n = self.values.shape[0]
        all_ratio = train_ratio + test_ratio + valid_ratio
        np.random.shuffle(self.values)
        self.train = np.asarray(self.values[:int((train_ratio*n)/all_ratio)])
        self.valid = np.asarray(self.values[int((train_ratio*n)/all_ratio):int(((train_ratio+valid_ratio)*n)/all_ratio)])
        self.test = np.asarray(self.values[int(((train_ratio+valid_ratio)*n)/all_ratio):])

    def get_distro(self, data):
        labels = data[:,-1]
        return np.divide(np.unique(labels, return_counts=True)[1], labels.size)

    def compute_entropy(self, data):
        distro = self.get_distro(data)
        log_distro = np.log2(distro)
        return 0 if np.count_nonzero(distro)<=1 else (-1 * distro * log_distro).sum() #np.sum(-1 * distro * log_distro)

    def information_gain(self, val, fea_id, entropy, data):
        smaller = data[np.where(data[:,fea_id]<=val)]
        greater = data[np.where(data[:,fea_id]>val)]
        prob_smaller = float(smaller.shape[0]) / (smaller.shape[0] + greater.shape[0])
        prob_greater = float(greater.shape[0]) / (smaller.shape[0] + greater.shape[0])
        s_entropy = self.compute_entropy(smaller)
        g_entropy = self.compute_entropy(greater)
        return (entropy - (prob_smaller*s_entropy + prob_greater*g_entropy), smaller, greater)

    def feature_choice(self, data):
        maxIG = (-1, 0, 0, [], [])
        thres, samples = self.features, data.shape[0]
        entropy = self.compute_entropy(data)
        for fea_id in range(self.features):
            for split_id in range(samples):
                inf_gain = self.information_gain(data[split_id, fea_id], fea_id, entropy, data)
                if maxIG[0] < inf_gain[0]:
                    maxIG = (inf_gain[0], split_id, fea_id, inf_gain[1], inf_gain[2])
        return maxIG[1:]

    def set_root_node(self, split_id, fea_id):
        self.root = Node()
        self.root.set_all(self.train[split_id][fea_id], fea_id, self.nodesCnt, self.train)
        self.nodesCnt += 1

    def setup_node(self, data, node, split_id, fea_id):
        node.set_all(data[split_id][fea_id], fea_id, self.nodesCnt, data)
        self.nodesCnt += 1

    def create_tree(self):
        split_id, fea_id, s, g = self.feature_choice(self.train)
        self.set_root_node(split_id, fea_id)
        que = queue.Queue()
        que.put((self.root, 'l', s, 1))
        que.put((self.root, 'r', g, 1))
        while not que.empty():
            node, side, data, depth = que.get()
            if depth > self.depth_threshold or not len(data):
                continue
            node.create_left() if side == 'l' else node.create_right()
            node = node.left if side == 'l' else node.right
            split_id, fea_id, s, g = self.feature_choice(data)
            self.setup_node(data, node, split_id, fea_id)
            que.put((node, 'l', s, depth+1))
            que.put((node, 'r', g, depth+1))

    def calculate_accuracy(self, result, predicted_result):
        return np.sum(result == predicted_result) / float(result.size)
    
    def set_next_node(self, t, fea_id):
        if t.left is None or t.right is None:
            return t.left if t.right is None else t.right
        else:
            return t.right if fea_id > t.threshold else t.left

    def evaluate_one_sample(self, node, sample):
        while True:
            if node.left is None and node.right is None:
                label = np.argmax(np.bincount(node.labels))
                return int(sample[-1]), int(label)
            else:
                node = self.set_next_node(node, sample[node.feature_id])

    def evaluate_tree(self, test_data):
        predicted_result, result = [], []
        for sample in test_data:
            t = self.root
            sample_label, pred_label = self.evaluate_one_sample(t, sample)
            result.append(sample_label)
            predicted_result.append(pred_label)
        result, predicted_result = np.asarray(result), np.asarray(predicted_result)
        accuracy = self.calculate_accuracy(result, predicted_result)
        return ("Accuracy obtained on test data is %f" % (accuracy*100), accuracy, result, predicted_result)

    def next_nodes_are_leaves(self, left_node, right_node):
        if left_node and not (left_node.left is None and left_node.right is None):
            return False
        if right_node and not (right_node.left is None and right_node.right is None):
            return False
        return True

    def get_nodes_with_leaves(self, node, nodes_with_leaves):
        if node is None or (node.left is None and node.right is None) or not node.prune_flag:
            return
        if self.next_nodes_are_leaves(node.left, node.right):
            nodes_with_leaves.append(node)
            return
        self.get_nodes_with_leaves(node.left, nodes_with_leaves)
        self.get_nodes_with_leaves(node.right, nodes_with_leaves)

    def cut_or_not(self, node, eps):
        info, acc, *_ = self.evaluate_tree(self.valid)
        left_node, right_node = node.left, node.right
        node.left, node.right = None, None
        info, acc2, *_ = self.evaluate_tree(self.valid) 
        if not (acc2 >= acc + eps):
            node.left, node.right = left_node, right_node
        node.prune_flag = False if acc2 < acc + eps else True
        return self.to_remove if acc2 >= acc + eps else self.to_preserve
    
    def prune_tree(self, eps = 0):
        while True:
            nodes_before_leaves = []
            self.get_nodes_with_leaves(self.root, nodes_before_leaves)
            if nodes_before_leaves == []:
                return
            nodes_with_leaves = np.asarray(nodes_before_leaves)
            func = np.vectorize(self.cut_or_not)
            result_array = func(nodes_with_leaves, eps)
            if not np.any(result_array == self.to_remove):
                return