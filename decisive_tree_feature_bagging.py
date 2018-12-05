import csv, random, math as m, queue, numpy as np
from Node import Node

class DecisiveTreeFeatureBagging:
    def __init__(self, file):
        self.file = file
        self.columns, self.used_in_tree = [], []
        self.weights, self.train, self.valid, self.test = {}, np.array([]), np.array([]), np.array([])
        self.features, self.root, self.nodesCnt, self.threshold = 0, 0, 0, 11
        self.to_preserve, self.to_remove = 0, 1
        self.root = None

    def set_dataset(self, train, valid, test):
        self.train, self.valid, self.test = train, valid, train
        self.features = 11
        self.used_in_tree = [False] * self.features

    def get_distro(self, data):
        labels = data[:,-1]
        return np.divide(np.unique(labels, return_counts=True)[1], labels.size)

    def compute_entropy(self, data):
        distro = self.get_distro(data)
        return 0 if np.count_nonzero(distro)<=1 else sum(-1*pi*m.log(pi, 2) for pi in distro)

    def information_gain(self, val, fea_id, entropy, data):
        smaller = data[np.where(data[:,fea_id]<=val)]
        greater = data[np.where(data[:,fea_id]>val)]
        prob_smaller = smaller.shape[0] / (smaller.shape[0] + greater.shape[0])
        prob_greater = greater.shape[0] / (smaller.shape[0] + greater.shape[0])
        s_entropy = self.compute_entropy(smaller)
        g_entropy = self.compute_entropy(greater)
        return (entropy - (prob_smaller*s_entropy + prob_greater*g_entropy), smaller, greater)

    def feature_choice_feature_bagging(self, data, used_already, avail_features):
        maxIG = (-1, 0, 0, [], [])
        thres, samples = self.features, data.shape[0]
        entropy = self.compute_entropy(data)
        for fea_id in avail_features:
            for split_id in range(samples):
                inf_gain = self.information_gain(data[split_id, fea_id], fea_id, entropy, data)
                if maxIG[0] < inf_gain[0]:
                    maxIG = (inf_gain[0], split_id, fea_id, inf_gain[1], inf_gain[2])
        used_already[maxIG[2]] = True
        return maxIG[1:]

    def set_root_node(self, split_id, fea_id):
        self.root = Node(self.used_in_tree)
        self.root.set_all(self.train[split_id][fea_id], fea_id, self.nodesCnt, self.train)
        self.nodesCnt += 1

    def setup_node(self, data, node, split_id, fea_id):
        node.used_feature(fea_id)
        node.set_all(data[split_id][fea_id], fea_id, self.nodesCnt, data)
        self.nodesCnt += 1

    def get_available_features(self, already_used_features):
        avail_features = [i for i,v in enumerate(already_used_features) if not v]
        formula = m.ceil(m.sqrt(len(avail_features)))
        sample_to_choose = formula if formula > 0 else 1
        return np.random.choice(avail_features, sample_to_choose, replace=False)
    
    def get_available_features2(self, already_used_features):
        formula = m.ceil(m.sqrt(11))
        sample_to_choose = formula if formula > 0 else 1
        return np.random.choice(list(range(0,11)), sample_to_choose, replace=False)

    def create_tree_feature_bagging(self):
        avail_features = self.get_available_features2(self.used_in_tree)
        split_id, fea_id, s, g = self.feature_choice_feature_bagging(self.train,
            self.used_in_tree, avail_features)
        self.set_root_node(split_id, fea_id)
        que = queue.Queue()
        que.put((self.root, 'l', s, 1))
        que.put((self.root, 'r', g, 1))
        while not que.empty():
            node, side, data, depth = que.get()
            if depth > self.threshold or len(data) == 0:
                continue
            node.create_left(node.used_already) if side == 'l' else node.create_right(node.used_already)
            node = node.left if side == 'l' else node.right
            avail_features = self.get_available_features2(node.used_already)
            split_id, fea_id, s, g = self.feature_choice_feature_bagging(data,
                node.used_already, avail_features)
            self.setup_node(data, node, split_id, fea_id)
            que.put((node, 'l', s, depth+1))
            que.put((node, 'r', g, depth+1))

    def calculate_accuracy(self, result, predicted_result):
        return np.sum(result == predicted_result)/result.size
    
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
        return ("Accuracy obtained on test data is %f" % accuracy, accuracy, result, predicted_result)
