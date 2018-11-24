import sys, csv, random, math, queue
from copy import deepcopy
from igraph import *

class Node:
    def __init__(self):
        # brakuje tutaj labela, jesli node jest lisciem to ktorys numer od 1 do 10, jesli nie to -1
        # niekoniecznie musi byc jednoznacznosc, wiec fajnie zrobic losowanko, gdy mozliwosc jest wiecej
        # na przyklad gdy depth nie pozwala na wieksza dokladnosc
        self.threshold, self.feature_id, self.node_id = -1, -1, -1
        self.used_already = []
        self.left, self.right = None, None
    def __repr__(self):
        return 'Node for %d feature' % self.feature_id
    def set_all(self, threshold, feature_id, node_id):
        self.threshold = threshold
        self.feature_id = feature_id
        self.node_id = node_id
    def is_set(self, id):
        return self.used_already[id]
    def used_feature(self, id):
        self.used_already[id] = True
    def create_left(self, used_already):
        self.left = Node()
        self.left.used_already = used_already
    def create_right(self, used_already):
        self.right = Node()
        self.right.used_already = used_already
    def get_used(self):
        return self.used_already

class Decisive_Tree:
    def __init__(self, file):
        self.file = file
        self.columns, self.values, self.used_in_tree = [], [], []
        self.train, self.valid, self.test = [], [], []
        self.features, self.root, self.nodesCnt = 0, 0, 0
        self.root = Node()

    def create_dataset(self, train_ratio, valid_ratio, test_ratio):
        with open(self.file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 200:
                    break
                line = row[0].split(';')
                if line_count == 0:
                    self.columns = [str(el.replace('"', '')) for el in line]
                    line_count += 1
                    self.features = len(line) - 1
                    self.used_in_tree = [False] * self.features
                else:
                    line = [float(el) for el in line]
                    self.values.append(line)
                    line_count += 1
        
        n = len(self.values)
        all_ratio = train_ratio + test_ratio + valid_ratio
        random.shuffle(self.values)
        self.train = self.values[:int((train_ratio*n)/all_ratio)]
        self.valid = self.values[int((train_ratio*n)/all_ratio):int(((train_ratio+valid_ratio)*n)/all_ratio)]
        self.test = self.values[int(((train_ratio+valid_ratio)*n)/all_ratio):]

    def get_labels(self, data):
        number_of_samples = len(data)
        id_label = self.features
        labels = [data[i][id_label] for i in range(number_of_samples)]
        unique_labels = sorted(list(set(labels)))
        return (id_label, labels, unique_labels)

    def compute_labels_distribution(self, labels, unique_labels):
        return [labels.count(el)/len(labels) for el in unique_labels]

    def compute_entropy(self, data):
        id_label, labels, unique_labels = self.get_labels(data)
        labels_distro = self.compute_labels_distribution(labels, unique_labels)
        return sum(-1*pi*math.log(pi, 2) for pi in labels_distro)

    def information_gain(self, f_id, t_ex_id, entropy, data):
        len("data size is ", len(data))
        smaller = [data[i] for i in range(len(data)) if data[t_ex_id][f_id] >= data[i][f_id]]
        greater = [data[i] for i in range(len(data)) if data[t_ex_id][f_id] < data[i][f_id]]
        # mysle, ze one nie powinny zwracac tylko indeksow... lepiej cale podtablice... ale czy to bedzie wydajne?
        prob_smaller = len(smaller)/(len(smaller) + len(greater))
        prob_greater = len(greater)/(len(smaller) + len(greater))
        s_entropy = self.compute_entropy(smaller)
        g_entropy = self.compute_entropy(greater)
        return (entropy - (prob_smaller*s_entropy - prob_greater*g_entropy), smaller, greater)

    def feature_choice(self, entropy, data, used_already):
        maxIG = (-1, 0, 0, [], [])
        for i in range(self.features):
            if not used_already[i]:
                for j,v in enumerate(data):
                    inf_gain = self.information_gain(i, j, entropy, data)
                    if maxIG[0] < inf_gain[0]:
                        maxIG = (inf_gain[0], i, j, inf_gain[1], inf_gain[2])
        used_already[maxIG[1]] = True
        return maxIG # tupla bez pierwszego elementu to maxIG[1:] ?

    def create_tree(self):
        threshold = 5
        entropy = self.compute_entropy(self.train)
        maxIG, fea_id, split_id, s, g = self.feature_choice(entropy, self.train, self.used_in_tree)
        print(fea_id, split_id)
        self.used_in_tree[fea_id] = True
        self.root.set_all(self.train[split_id][fea_id], fea_id, self.nodesCnt)
        que = queue.Queue()
        self.root.create_left(deepcopy(self.used_in_tree))
        self.root.create_right(deepcopy(self.used_in_tree))
        que.put((self.root.left, s, 1)) # qsize(), empty(), get()
        que.put((self.root.right, g, 1))
        while not que.empty():
            node, data, depth = que.get()
            if depth > threshold: 
                continue
            maxIG, fea_id, split_id, s, g = self.feature_choice(entropy, data, node.get_used())
            node.used_feature(fea_id)
            self.nodesCnt += 1
            node.set_all(data[split_id][fea_id], fea_id, self.nodesCnt)
            # sprawdzic gdzie jest wykorzystywane used_in_tree i podmienic na node.used_already
            if len(s):
                node.create_left(deepcopy(node.get_used()))
                que.put((node.left, s, depth+1))
            if len(g):
                node.create_right(deepcopy(node.get_used()))
                que.put((node.right, g, depth+1))

    def evaluate_tree(self):
        pass

    def prune_tree(self):
        pass

def bfs(node, g):
    if node == None:
        return
    if node.left != None and node.left.node_id != -1:
        print(node.node_id, node.left.node_id)
        g.add_edges([(node.node_id, node.left.node_id)])
        bfs(node.left, g)
    if node.right != None and node.right.node_id != -1:
        print(node.node_id, node.right.node_id)
        g.add_edges([(node.node_id, node.right.node_id)])
        bfs(node.right, g)

def draw_decision_tree(tree):
    g = Graph()
    print("na poczatku ", tree.nodesCnt)
    #g.add_vertices(tree.nodesCnt)
    g.add_vertices(45)
    bfs(tree.root, g)
    layout = g.layout("kamada_kawai")
    plot(g, layout = layout)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit()
    tree = Decisive_Tree(str(sys.argv[1]))
    print("Enter train_ratio, test_ratio and valid_ratio for data:")
    train_ratio, valid_ratio, test_ratio = map(int, sys.stdin.readline().split())
    tree.create_dataset(train_ratio, valid_ratio, test_ratio)
    tree.create_tree()
    #draw_decision_tree(tree)