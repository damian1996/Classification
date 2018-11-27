import csv, random, math, queue, numpy as np
from Node import Node

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
            self.features = len(splitted_lines[0]) - 1
            self.used_in_tree = [False] * self.features
            for line in splitted_lines[1:5000]:
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
        return 0 if np.count_nonzero(distro)<=1 else sum(-1*pi*math.log(pi, 2) for pi in distro)

    def information_gain(self, val, fea_id, entropy, data):
        smaller = data[np.where(data[:,fea_id]<=val)]
        greater = data[np.where(data[:,fea_id]>val)]
        #column = data[:,fea_id]
        #smaller1 = column[np.where(column <= val)]
        #greater1 = column[np.where(column > val)] # np.unique?
        #smaller = data[np.in1d(data[:, fea_id], smaller1)]
        #greater = data[np.in1d(data[:, fea_id], greater1)]
        prob_smaller = smaller.shape[0] / (smaller.shape[0] + greater.shape[0])
        prob_greater = greater.shape[0] / (smaller.shape[0] + greater.shape[0])
        s_entropy = self.compute_entropy(smaller)
        g_entropy = self.compute_entropy(greater)
        return (entropy - (prob_smaller*s_entropy + prob_greater*g_entropy), smaller, greater)

    def feature_choice(self, data, used_already):
        maxIG = (-1, 0, 0, [], [])
        thres, samples = self.features, data.shape[0]
        entropy = self.compute_entropy(data)
        for fea_id in range(self.features):
            if not used_already[fea_id]:
                '''
                inf_gain = np.vectorize(lambda y: (y[0], y[1])) #(self.information_gain(y[1], fea_id, entropy, data), y[0])
                result_inf_gain = inf_gain(np.dstack((np.arange(samples), data[:,fea_id])))
                print(result_inf_gain)
                ig = np.amax(result_inf_gain, axis=0)
                if maxIG[0] < ig[0][0]:
                    maxIG = (ig[0][0], ig[1], fea_id, ig[0][1], ig[0][2])
                '''
                for split_id in range(samples):
                    inf_gain = self.information_gain(data[split_id, fea_id], fea_id, entropy, data)
                    if maxIG[0] < inf_gain[0]:
                        maxIG = (inf_gain[0], split_id, fea_id, inf_gain[1], inf_gain[2])
        '''
        for split_id, fea_id in np.ndindex(data.shape): # byc moze po prostu podwojna petla bedzie miala lepszy performance
            if fea_id < thres and not used_already[fea_id]: # wtedy mniej tych ifow, potem zmienie
                inf_gain = self.information_gain(data[split_id, fea_id], split_id, fea_id, entropy, data)
                if maxIG[0] < inf_gain[0]:
                    maxIG = (inf_gain[0], split_id, fea_id, inf_gain[1], inf_gain[2])
        '''
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

    def create_tree(self):
        split_id, fea_id, s, g = self.feature_choice(self.train, self.used_in_tree)
        self.set_root_node(split_id, fea_id)
        que = queue.Queue()
        que.put((self.root, 'l', s, 1))
        que.put((self.root, 'r', g, 1))
        while not que.empty():
            node, side, data, depth = que.get()
            if depth > self.threshold or not len(data):
                continue
            node.create_left(node.get_used()) if side == 'l' else node.create_right(node.get_used())
            node = node.left if side == 'l' else node.right
            split_id, fea_id, s, g = self.feature_choice(data, node.get_used())
            self.setup_node(data, node, split_id, fea_id)
            que.put((node, 'l', s, depth+1))
            que.put((node, 'r', g, depth+1))

    def calculate_accuracy(self, result, predicted_result):
        return np.sum(result == predicted_result)/result.size
    
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
                    label = np.argmax(np.bincount(t.labels))                    
                    result.append(int(sample[-1]))
                    predicted_result.append(int(label))
                    break
                else:
                    t = self.set_node(t, sample[t.feature_id])
        result, predicted_result = np.asarray(result), np.asarray(predicted_result)
        return ("Accuracy obtained on test data is %f" % self.calculate_accuracy(result, predicted_result), result, predicted_result)

    def prune_tree(self):
        pass