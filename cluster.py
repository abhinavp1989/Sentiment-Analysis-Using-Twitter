"""
cluster.py
"""
import networkx as nx
import matplotlib.pyplot as plt
import pickle

'''Girvan Newman Networkx
https://networkx.github.io/documentation/networkx-2.0/_modules/networkx/algorithms/community/centrality.html
'''

def girvan_newman(G, most_valuable_edge=None):
    """
     Parameters
    ----------
    G : NetworkX graph

    most_valuable_edge : function
        Function that takes a graph as input and outputs an edge. The
        edge returned by this function will be recomputed and removed at
        each iteration of the algorithm.

        If not specified, the edge with the highest
        :func:`networkx.edge_betweenness_centrality` will be used.

    Returns
    -------
    iterator
        Iterator over tuples of sets of nodes in `G`. Each set of node
        is a community, each tuple is a sequence of communities at a
        particular level of the algorithm.
    """
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    if most_valuable_edge is None:
        def most_valuable_edge(G):
            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)
    g = G.copy().to_undirected()
    g.remove_edges_from(g.selfloop_edges())
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)



def _without_most_central_edges(G, most_valuable_edge):
    """
    Returns the connected components of the graph that results from
    repeatedly removing the most "valuable" edge in the graph.

    `G` must be a non-empty graph. This function modifies the graph `G`
    in-place; that is, it removes edges on the graph `G`.

    `most_valuable_edge` is a function that takes the graph `G` as input
    (or a subgraph with one or more edges of `G` removed) and returns an
    edge. That edge will be removed and this process will be repeated
    until the number of connected components in the graph increases.

    """
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components

def main():

    graph = nx.read_edgelist("cluster_data.txt",delimiter=",")
    components = [c for c in nx.connected_component_subgraphs(graph)]
    f2 =[]
    for c1 in components:
        k = girvan_newman(c1)
        f2.append(tuple(sorted(c) for c in next(k)))
    i = 0
    clust_avg = []
    for i in range(len(f2)):
        for j in range(0, 2):
            clust_avg.append(len(f2[i][j]))
    pickle.dump(f2, open('cluster.pkl', 'wb'))
    plt.figure(figsize=(15,12))  
    nx.draw(graph,node_color='darkturquoise',width=.9,alpha=0.2)
    plt.savefig("Cluster.png")
    f = open("cluster.txt", "w+")
    f.write("NUMBER OF COMMUNITIES DISCOVERED : %d\n" % (len(components)))
    f.write("AVERAGE NUMBER OF USERS PER COMMUNITY : %d\n" % (sum(clust_avg) / len(clust_avg)))
    f.close()
    
if __name__ == '__main__':
    main()

