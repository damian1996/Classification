import csv, random, math, queue, numpy as np
from Node import Node
from decisive_tree import Decisive_Tree

class RandomForest:
    def __init__(self, file):
        self.file = file
        self.card_forest, self.samples_per_tree = 20, 500
        self.feature_bagging = np.array((self.samples_per_tree, self.card_forest))
        self.valid, self.test = np.array([])
        self.forest = [Decisive_Tree(file)]*self.card_forest
        self.bags = np.empty([self.card_forest, self.samples_per_tree])

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

    def extract_datasets(self, train_ratio, valid_ratio, test_ratio):
        n = self.values.shape[0]
        all_ratio = train_ratio + test_ratio + valid_ratio
        np.random.shuffle(self.values)
        self.train = np.asarray(self.values[:int((train_ratio*n)/all_ratio)])
        self.valid = np.asarray(self.values[int((train_ratio*n)/all_ratio):int(((train_ratio+valid_ratio)*n)/all_ratio)])
        self.test = np.asarray(self.values[int(((train_ratio+valid_ratio)*n)/all_ratio):])

    def create_tree(self, tree):
        smaller_train = np.random.choice(self.train, self.samples_per_tree)
        tree.set_dataset(smaller_train, self.valid, self.test)
        tree.create_tree()
        tree.prune_tree(eps=0.0003)

    def create_random_forest(self, trees):
        for i in range(self.card_forest):
            self.create_tree(self.forest[i])
    
    def evaluate_random_forest(self):
        # po jednym przykladzie czy wszystkie na raz?
'''
• create_random_forest: budujemy las losowy korzystając z drzew decyzyjnych i tree bagging.
• evaluate_random_forest: uruchamiamy wytrenowany klasyfikator na zbiorze przykładów. Procedura
powinna zwrócić klasy, jakie przypisał klasyfikator oraz skuteczność (accuracy).
'''