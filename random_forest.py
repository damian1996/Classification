import csv, random, math as m, queue, numpy as np
from Node import Node
from decisive_tree_feature_bagging import DecisiveTreeFeatureBagging
from Draw import draw_decision_tree
from joblib import Parallel, delayed

class RandomForest:
    def __init__(self, file):
        self.file = file
        self.card_forest, self.samples_per_tree = 100, 2200
        self.feature_bagging = np.array((self.samples_per_tree, self.card_forest))
        self.values, self.train, self.valid, self.test = np.array([]), np.array([]), np.array([]), np.array([])
        self.forest = [DecisiveTreeFeatureBagging(file) for _ in range(self.card_forest)]
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
        self.extract_datasets(train_ratio, valid_ratio, test_ratio)

    def extract_datasets(self, train_ratio, valid_ratio, test_ratio):
        n = self.values.shape[0]
        all_ratio = train_ratio + test_ratio + valid_ratio
        np.random.shuffle(self.values)
        self.train = np.asarray(self.values[:int((train_ratio*n)/all_ratio)])
        self.valid = np.asarray(self.values[int((train_ratio*n)/all_ratio):int(((train_ratio+valid_ratio)*n)/all_ratio)])
        self.test = np.asarray(self.values[int(((train_ratio+valid_ratio)*n)/all_ratio):])

    def create_tree(self, tree):
        smaller_train = self.train[np.random.choice(list(range(self.train.shape[0])),
            self.samples_per_tree, replace=True)]
        tree.set_dataset(smaller_train, self.valid, self.test)
        tree.create_tree_feature_bagging()

    def create_random_forest(self):
        for i in range(self.card_forest):
            print(i)
            self.create_tree(self.forest[i])
            self.evaluate_random_forest(i+1)
            #draw_decision_tree(self.forest[i])
    
    def calculate_accuracy(self, result, predicted_result):
        return np.sum(result == predicted_result)/result.size

    def two_most_frequent_probs(self, arr):
        bc = np.bincount(arr)
        return bc[-1]/len(arr), bc[-2]/len(arr)

    def evaluate_random_forest(self, nr_trees):
        result, pred_result = [], []
        thr_better, fr_better = 0, 0
        labels = self.test[:,-1]
        unique, counts = np.unique(labels, return_counts=True)
        pred_good = {k:0 for k in unique}
        pred_bad = {k:0 for k in unique}
        print("All labels ", dict(zip(unique, counts)))
        for i, sample in enumerate(self.test):
            res_for_sample = []
            for j in range(nr_trees): # self.card_forest
                t = self.forest[j].root
                real, pred = self.forest[j].evaluate_one_sample(t, sample)
                res_for_sample.append(pred)
            prob = res_for_sample.count(real)/len(res_for_sample)
            arr = np.asarray(res_for_sample)
            first, second = self.two_most_frequent_probs(arr)
            fr = np.argmax(np.bincount(arr))
            thr = round(np.sum(arr)/arr.size)
            if fr == real and thr != real:
                fr_better += 1
            elif fr != real and thr == real:
                thr_better += 1
            result.append(real)
            pred_result.append(fr)
            if real == fr:
                pred_good[real] += 1
            else:
                pred_bad[real] += 1
            #pred_result.append(thr)
            #pred_result.append(np.argmax(np.bincount(np.asarray(res_for_sample))))
        print("First measure better {} times".format(fr_better))
        print("Third measure better {} times".format(thr_better))
        print("Good predictions ", pred_good)
        print("Bad predictions ", pred_bad)
        result, predicted_result = np.asarray(result), np.asarray(pred_result)
        accuracy = self.calculate_accuracy(result, predicted_result)*100
        print("Accuracy obtained on test data is %f" % accuracy)
        return ("Accuracy obtained on test data is %f" % accuracy, accuracy, result, predicted_result)