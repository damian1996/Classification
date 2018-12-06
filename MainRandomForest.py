import sys
from decisive_tree_feature_bagging import DecisiveTreeFeatureBagging
from Draw import draw_decision_tree, draw_plot_for_labels
from random_forest import RandomForest

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit()
    results_sum, required = 0.0, 0.500000
    samples, better_than_required = 10, 0
    custom_split = False
    
    print("Do you want to enter proportions?\nDefault = 0\nCustom = 1")
    if int(sys.stdin.readline()):
            print("Enter train_ratio, valid_ratio and test_ratio for data:")
            train_ratio, valid_ratio, test_ratio = map(int, sys.stdin.readline().split())
            custom_split = True
    
    rf = RandomForest(sys.argv[1])
    rf.create_dataset() if not custom_split else rf.create_dataset(train_ratio, valid_ratio, test_ratio)
    rf.create_random_forest()
    info, accuracy, res_labels, predicted_labels = rf.evaluate_random_forest(rf.card_forest)
    print(info)

