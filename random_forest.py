import csv, random, math as m, queue, numpy as np
from Node import Node
from decisive_tree_feature_bagging import DecisiveTreeFeatureBagging

class RandomForest:
    def __init__(self, file):
        self.file = file
        self.card_forest, self.samples_per_tree = 30, 900
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
            self.samples_per_tree, replace=False)]
        tree.set_dataset(smaller_train, self.valid, self.test)
        tree.create_tree_feature_bagging()
        tree.prune_tree(eps=0.0003)

    def create_random_forest(self):
        for i in range(self.card_forest):
            self.create_tree(self.forest[i])
    
    def calculate_accuracy(self, result, predicted_result):
        return np.sum(result == predicted_result)/result.size

    def evaluate_random_forest(self):
        result, pred_result = [], []
        thr_better, fr_better = 0, 0
        for i, sample in enumerate(self.test):
            res_for_sample = []
            for j in range(self.card_forest):
                t = self.forest[j].root
                real, pred = self.forest[j].evaluate_one_sample(t, sample)
                res_for_sample.append(pred)
            #print("Predicted results ", res_for_sample)
            #print("Real result", real)
            prob = res_for_sample.count(real)/len(res_for_sample)
            arr = np.asarray(res_for_sample)
            fr = np.argmax(np.bincount(arr))
            thr = round(np.sum(arr)/arr.size)
            #print("Probability real ({}) result in predicted results is {}".format(real, prob))
            if fr == real and thr != real:
                print("Max_frequent better for probability ", prob)
                fr_better += 1
            elif fr != real and thr == real:
                #print("Round mean better for probability ", prob)
                thr_better += 1
            #print("max_freq pred = {}, round_mean pred = {}".format(fr, thr))
            result.append(real)
            pred_result.append(thr)
            #pred_result.append(np.argmax(np.bincount(np.asarray(res_for_sample))))
        print("First measure better {} times".format(fr_better))
        print("Third measure better {} times".format(thr_better))
        result, predicted_result = np.asarray(result), np.asarray(pred_result)
        accuracy = self.calculate_accuracy(result, predicted_result)
        return ("Accuracy obtained on test data is %f" % accuracy, accuracy, result, predicted_result)