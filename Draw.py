import Node
import matplotlib.pyplot as plt
from igraph import Graph, plot

def bfs(node, g, tree):
    if node == None:
        return
    #g.vs[node.node_id]["name"] = "%s" % (tree.columns[node.feature_id])
    g.vs[node.node_id]["name"] = "%d" % (node.node_id)    
    #if node.left is not None and node.right is not None:
    #    print(node.left.node_id, " ", node.right.node_id)
    if node.left != None and node.left.node_id != -1:
        #g.vs[node.node_id]["name"] = "%s" % (tree.columns[node.feature_id])
        g.add_edges([(node.node_id, node.left.node_id)])
        bfs(node.left, g, tree)
    if node.right != None and node.right.node_id != -1:
        #g.vs[node.node_id]["name"] = "%s" % (tree.columns[node.feature_id])
        g.add_edges([(node.node_id, node.right.node_id)])
        bfs(node.right, g, tree)

def draw_decision_tree(tree):
    g = Graph()
    g.add_vertices(tree.nodesCnt)
    bfs(tree.root, g, tree)
    g.vs["label"] = g.vs["name"]
    layout = g.layout("kamada_kawai")
    plot(g, layout = layout)

def draw_plot_for_labels(result, res_labels, predicted_labels):
    cnt_rl = {int(l):res_labels.count(l) for l in sorted(list(set(res_labels)))}
    cnt_pl = {int(l):predicted_labels.count(l) for l in sorted(list(set(predicted_labels)))}
    plt.plot([i for i in range(12)], [0 if k not in cnt_rl.keys() else cnt_rl[k] for k in range(12)])#, 'ro')
    plt.plot([i for i in range(12)], [0 if k not in cnt_pl.keys() else cnt_pl[k] for k in range(12)])#, 'bs')
    plt.show()