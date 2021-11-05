# by Huaiyu-TAN

# m: initial number of nodes
# n: final number of nodes
# k: nodes degree

import networkx as nx
import numpy as np
from time import time
    
class BA_sfnet:
    
    def __init__(self, m, n, k):
        self.m = m; self.n = n; self.k = k
        self.G = nx.complete_graph(self.m)
        self.new_node = self.m
    def choice_node(self):
        # print("最初节点有 {} 个".format(len(self.G.nodes)))
        nodes_probs = []
        for node in self.G.nodes():
            nodes_probs.append(self.G.degree(node) / (2 * len(self.G.edges()))) # 连边概率
        # print("连边概率：{}".format(nodes_probs))
        random_prob_node = np.random.choice(self.G.nodes(), p = nodes_probs) # 以连边概率p从全部节点中选择一个与新节点相连
        # print("随机区间内选择节点 {} 号".format(random_prob_node))
        return random_prob_node

    def add_edge(self): # 采用递归
        if len(self.G.edges()) == 0: # 基线条件
            random_prob_node = 0
        else:
            random_prob_node = BA_sfnet.choice_node(self)
        new_edge = (random_prob_node, self.new_node)
        if new_edge in self.G.edges():
            self.G.add_edge
        else:
            self.G.add_edge(self.new_node, random_prob_node) # 加入新边
            # print("加入边: ({} {})".format(self.new_node, random_prob_node)) # new_node+1显示错位
            
    def sfnet(self):
        count = 0
        start = time()
        for i in range(self.n - self.m):
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<第 {} 轮模拟>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>".format(count + 1))
            self.G.add_node(self.m + count)
            # print("添加第 {} 个节点".format(self.m + count + 1))
            count += 1
            for j in range(0, self.k):
                BA_sfnet.add_edge(self)
            self.new_node += 1
        end = time()
        
        # python排序从0开始，这里统一加1
        list_edges = list(self.G.edges)
        # edges_point1 = [list(G.edges)[i][0] + 1 for i in range(len(list_edges))]
        # edges_point2 = [list(G.edges)[i][1] + 1 for i in range(len(list_edges))]

        # 储存为邻接矩阵
        nebmatrix = np.zeros(shape = (self.n, self.n))
        for i in range(len(list_edges)):
            nebmatrix[ list_edges[i][0], list_edges[i][1] ] = 1
            nebmatrix[ list_edges[i][1], list_edges[i][0] ] = 1
        # np.savetxt('Adjacent_Matrix.csv', nebmatrix, delimiter = ',')
        print(nebmatrix)
        # print("\n\n 最终{}个节点的连边情况为：\n\n {} \n".format(self.n, self.G.edges()))
        # print("\n 系统内存在的节点：\n\n{}\n".format(self.G.nodes()))
        print("\n \n {}轮用时共为{}秒".format(self.n ,end - start))