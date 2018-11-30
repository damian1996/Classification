import csv, random, math, queue, numpy as np
from Node import Node
from decisive_tree import Decisive_Tree

class RandomForest:
    def __init__(self):
        self.card_forest = 10
        self.samples_per_tree = 500
        self.feature_bagging = np.array((self.samples_per_tree, self.card_forest))

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
        
    def create_tree(self):
        pass
    def create_random_forest(self, trees):
        pass
    def evaluate_random_forest(self):
        pass
'''
• modyfikacja procedury create_tree tak, by pozwalała na użycie feature bagging (patrz wiki) podczas
tworzenia drzewa.
• create_random_forest: budujemy las losowy korzystając z drzew decyzyjnych i tree bagging.
• evaluate_random_forest: uruchamiamy wytrenowany klasyfikator na zbiorze przykładów. Procedura
powinna zwrócić klasy, jakie przypisał klasyfikator oraz skuteczność (accuracy).
'''