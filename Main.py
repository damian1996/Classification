import sys
from decisive_tree import Decisive_Tree

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit()
    for i in range(5):
        tree = Decisive_Tree(str(sys.argv[1]))
        '''
        print("Do you want to enter proportions?\nDefault = 0\nCustom = 1")
        if map(int, sys.stdin.readline()):
            print("Enter train_ratio, test_ratio and valid_ratio for data:")
            train_ratio, valid_ratio, test_ratio = map(int, sys.stdin.readline().split())
            tree.create_dataset(train_ratio, valid_ratio, test_ratio)
        else:
            print("Skipped")
            tree.create_dataset()
        '''
        tree.create_dataset()
        tree.create_tree()
        result, res_labels, predicted_labels = tree.evaluate_tree()
        #draw_plot_for_labels(result, res_labels, predicted_labels)
        #draw_decision_tree(tree)
        print(result)