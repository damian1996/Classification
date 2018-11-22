import sys, csv, random

class Decisive_Tree:
    def __init__(self, file):
        self.file = file
        self.columns = []
        self.values = []
        self.train = []
        self.validation = []
        self.test = []

        with open(self.file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                line = row[0].split(';')
                if line_count == 0:
                    self.columns = [str(el.replace('"', '')) for el in line]
                    print(self.columns)
                    line_count += 1
                else:
                    line = [float(el) for el in line]
                    self.values.append(line)
                    line_count += 1

    def create_dataset(self, train_ratio, valid_ratio, test_ratio):
        n = len(self.values)
        all_ratio = train_ratio + test_ratio + valid_ratio
        random.shuffle(self.values)
        self.train = self.values[:int((train_ratio*n)/all_ratio)]
        self.valid = self.values[int((train_ratio*n)/all_ratio):int(((train_ratio+valid_ratio)*n)/all_ratio)]
        self.test = self.values[int(((train_ratio+valid_ratio)*n)/all_ratio):]

    def create_tree(self):
        pass

    def evaluate_tree(self):
        pass

    def prune_tree(self):
        pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit()
    tree = Decisive_Tree(str(sys.argv[1]))
    print("Enter train_ratio, test_ratio and valid_ratio for data:")
    train_ratio, valid_ratio, test_ratio = map(int, sys.stdin.readline().split())
    tree.create_dataset(train_ratio, valid_ratio, test_ratio)