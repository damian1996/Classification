import sys
from decisive_tree import Decisive_Tree
from Draw import draw_decision_tree, draw_plot_for_labels

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit()
    results_sum, required = 0.0, 0.500000
    samples, better_than_required = 10, 0
    for i in range(samples):
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
        #draw_decision_tree(tree)
        tree.prune_tree(eps=0.0003)
        #draw_decision_tree(tree)
        info, accuracy, res_labels, predicted_labels = tree.evaluate_tree(tree.test)
        #draw_plot_for_labels(result, res_labels, predicted_labels)
        print(info)
        results_sum += accuracy
        if accuracy >= required:
            better_than_required += 1

    print("Mean result is equally to ", results_sum/samples)
    print("{} tries, {} better than required threshold".format(samples, better_than_required))
