import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

# G = nx.random_geometric_graph()

class Gen_RGG:
    def plot_G(G, nodesize):
        nx.draw(G,node_size = nodesize)
        plt.show()
    def find_average_k(G):
        t = 0
        for i in range(len(G)):
            t += G.degree(i)
        ave_k = t / len(G)
        return ave_k
    def Gen_process(n, k):
        T = 0
        for i in [i / 100000 for i in range(1, 100000)]:
            G = nx.random_geometric_graph(n, i)
            iter_ave_k = Gen_RGG.find_average_k(G)
            print("iter_round {} , p = {} , average_k = {}".format(T, i, iter_ave_k))
            if round(abs(iter_ave_k - k), 2) <= .001:
                nebmat = nx.to_numpy_matrix(G)
                np.savetxt('nebmat.csv', nebmat, delimiter=',')
                print("\n\n complete iteration !")
                return (nebmat, i, G)
            T += 1

if __name__ == "__main__":
    (nebmat, p, G) = Gen_RGG.Gen_process(1000, 3)
    Gen_RGG.plot_G(G, 35)