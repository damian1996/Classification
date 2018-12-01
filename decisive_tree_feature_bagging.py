import csv, random, math as m, queue, numpy as np
from Node import Node

class DecisiveTreeFeatureBagging:
    def __init__(self, file):
        self.file = file
        self.columns, self.used_in_tree = [], []
        self.values, self.train, self.valid, self.test = np.array([]), np.array([]), np.array([]), np.array([])
        self.features, self.root, self.nodesCnt, self.threshold = 0, 0, 0, 10
        self.to_preserve, self.to_remove = 0, 1
        self.root = None

    def create_dataset(self, train_ratio=5, valid_ratio=1, test_ratio=1):
        with open(self.file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            splitted_lines = [x[0].split(';') for x in csv_reader]
            self.columns = [str(el.replace('"', '')) for el in splitted_lines[0]]
            self.features = len(splitted_lines[0]) - 1
            self.used_in_tree = [False] * self.features
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
            if not used_already[fea_id]: # niepotrzebne na 99.99%
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
        sample_to_choose = m.ceil(m.sqrt(len(avail_features)))
        return np.random.choice(avail_features, sample_to_choose, replace=False)
   
    def create_tree_feature_bagging(self):
        avail_features = self.get_available_features(self.used_in_tree)
        split_id, fea_id, s, g = self.feature_choice_feature_bagging(self.train,
            self.used_in_tree, avail_features)
        self.set_root_node(split_id, fea_id)
        que = queue.Queue()
        que.put((self.root, 'l', s, 1))
        que.put((self.root, 'r', g, 1))
        while not que.empty():
            node, side, data, depth = que.get()
            if depth > self.threshold or not len(data):
                continue
            node.create_left(node.used_already) if side == 'l' else node.create_right(node.used_already)
            node = node.left if side == 'l' else node.right
            avail_features = self.get_available_features(node.used_already)
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
            '''
            while True:
                if t.left is None and t.right is None:
                    label = np.argmax(np.bincount(t.labels))                    
                    result.append(int(sample[-1]))
                    predicted_result.append(int(label))
                    break
                else:
                    t = self.set_next_node(t, sample[t.feature_id])
            '''
        result, predicted_result = np.asarray(result), np.asarray(predicted_result)
        accuracy = self.calculate_accuracy(result, predicted_result)
        return ("Accuracy obtained on test data is %f" % accuracy, accuracy, result, predicted_result)

    def next_nodes_are_leaves(self, left_node, right_node):
        if left_node and not (left_node.left is None and left_node.right is None):
            return False
        if right_node and not (right_node.left is None and right_node.right is None):
            return False
        return True

    def get_nodes_with_leaves(self, node, nodes_with_leaves):
        if node is None or (node.left is None and node.right is None):
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
        #print(eps, acc2, acc)
        if not (acc2 >= acc + eps):
            node.left, node.right = left_node, right_node
        return self.to_remove if acc2 >= acc + eps else self.to_preserve
    
    def count_nodes(self, node, cnt):
        cnt[0] += 1
        if node.left:
            self.count_nodes(node.left, cnt)
        if node.right:
            self.count_nodes(node.right, cnt)

    def prune_tree(self, eps = 0):
        print("Before {}".format(self.nodesCnt))
        while True:
            nodes_before_leaves = []
            self.get_nodes_with_leaves(self.root, nodes_before_leaves)
            nodes_with_leaves = np.asarray(nodes_before_leaves)
            func = np.vectorize(self.cut_or_not)
            result_array = func(nodes_with_leaves, eps)
            if not self.to_remove in result_array:
                return
        cnt = [0, 0]
        count_nodes(self.root, cnt)
        print("After {}".format(cnt[0]))