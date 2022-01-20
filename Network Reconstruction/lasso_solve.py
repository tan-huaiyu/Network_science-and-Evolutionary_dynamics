# by Huaiyu HAN
# 2021-2-1

import numpy as np
import cvxpy as cp
import math, random
import networkx as nx
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

start1 = time()
# Gen Network
class Gen_Net:
    def basf(n: int, k: int):
        G_sf = nx.barabasi_albert_graph(n, k)
        return G_sf
    def ER(n: int, p: float):
        G_er = nx.random_graphs.erdos_renyi_graph(n, p)
        return G_er
    def SW(n: int, k: int, p: float):
        G_sw = nx.random_graphs.watts_strogatz_graph(n, k, p)
        return G_sw

# Define Useful Tools
class Use_Tools:
    def chunk(lst, size): # make selected array sort with rule size
        return list(map(lambda x: lst[x * size: x * size + size], list(range(0, math.ceil(len(lst) / size)))))
    def payoff(si, M, sj) -> float: # si*M*sj' and input is np.array
        phi = np.matmul(np.matmul(si, M), sj.T)
        return phi

# Define Game Structure
class Game:
    def PDG(b: float):
        M = np.array([[1, 0], [b, 0]])  # b = 1.2; M[0][0]=1; M[0][1]=0; M[1][0]=1.2; M[1][1]=0
        return M
    def SDG(r: float):
        M = np.array([[1, 1 - r], [1 + r, 0]])
        return M

# Initalize Strategies Set
G = Gen_Net.basf(50, 3) # average degree <k> = 6: k = 3
node_list = np.array(list(G.nodes)) + 1 # all nodes
edge_list = np.array(G.edges) + 1 # linkage's edges
nebmat = nx.to_numpy_matrix(G) # Adjacent Matrix of G
np.savetxt('A_M.csv', nebmat, delimiter=',')

s_set = [] # every node which choose C or D with equal prob
for _ in range(len(node_list)):
    temp_rand = np.random.rand(1)
    if temp_rand > 1/2: s_set.append((0, 1))
    else: s_set.append((1, 0)) # strategies set of #N nodes

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
pre_fail = []
for L in range(50, 500, 50):
    # Monte-Carlo Simulation
    Phi = []; Y_test = np.zeros([L, G.number_of_nodes()])
    M = Game.PDG(1.2) # MC Rounds
    for _ in range(G.number_of_nodes()): # Store Phi
        Phi.append(np.zeros([L, G.number_of_nodes()]))

    for l in range(L):
        for f in range(G.number_of_nodes()): # asynchronous updating
            temp_payoff, neb_temp_payoff = [], []
            focal_player = random.sample(list(node_list), 1)[0]
            # print("\n", focal_player)

            # find focal's neb and calculate payoff of focal player
            rdn_neb = []
            for i in range(len(edge_list)):
                if edge_list[i][0] != focal_player:
                    if edge_list[i][1] == focal_player:
                        rdn_neb.append(edge_list[i][0])
                        temp_payoff.append(Use_Tools.payoff(np.array(s_set[edge_list[i][1] - 1]), M, np.array(s_set[edge_list[i][0] - 1])))
                else:
                    rdn_neb.append(edge_list[i][1])
                    temp_payoff.append(Use_Tools.payoff(np.array(s_set[edge_list[i][0] - 1]), M, np.array(s_set[edge_list[i][1] - 1])))
            # print("\n", rdn_neb)
            # print("\n", temp_payoff)
            f_i = np.sum(temp_payoff)

            # find neb's neb and calculate payoff of player's neb
            select_neb = random.sample(rdn_neb, 1)[0]
            for i in range(len(edge_list)):
                if edge_list[i][0] != select_neb:
                    if edge_list[i][1] == select_neb:
                        neb_temp_payoff.append(Use_Tools.payoff(np.array(s_set[edge_list[i][1] - 1]), M, np.array(s_set[edge_list[i][0] - 1])))
                else:
                    neb_temp_payoff.append(Use_Tools.payoff(np.array(s_set[edge_list[i][0] - 1]), M, np.array(s_set[edge_list[i][1] - 1])))
            # print("\n", select_neb)
            # print("\n", neb_temp_payoff)
            f_j = np.sum(neb_temp_payoff)

            # Fermi updating
            # fi_lambda = np.sum(nebmat[focal_player - 1])  # normlization
            # fj_lambda = np.sum(nebmat[select_neb - 1])
            # w = 1 / (1 + math.exp(((f_i / fi_lambda) - (f_j / fj_lambda)) / 0.1)) # k = .1
            w = 1 / (1 + math.exp((f_i - f_j) / 0.1))  # k = .1
            if np.random.rand(1) >= w: s_set[focal_player - 1] = s_set[select_neb - 1]
            print("\n th {} Rounds of Fermi updating:\n {}".format(f, s_set))
            Y_test[l][focal_player - 1] = f_i
        # find Phi
        for m in range(G.number_of_nodes()): # #(m) matrix
            for c in range(G.number_of_nodes()): # #(col) of #(m)
                Phi[m][l][c] = Use_Tools.payoff(np.array(s_set[m]), M, np.array(s_set[c]))
        print("\n\n th {} Rounds of MC Simulation".format(l))
    end1 = time()

    # find Y
    Y1 = []
    for i in range(G.number_of_nodes()):
        Y1.append(np.matmul(Phi[i], nebmat[i, :].reshape(G.number_of_nodes(), 1)))
    Y = np.matrix(np.zeros((L, G.number_of_nodes())))
    for i in range(G.number_of_nodes()):
        Y[:,i] = Y1[i]

    # check weather Y and Phi are correct
    # Y1_hat = np.matmul(np.matrix(Phi[0]),nebmat[:,0])
    # Y1 = Y[:,0].reshape(100, 1)
    # Y1 == Y1_hat


    # Lasso solve
    start2 = time()
    X_bar = []
    for i in range(G.number_of_nodes()):
        X = cp.Variable(G.number_of_nodes())
        # obj = cp.Minimize(cp.multiply((1 / (2 * L)), cp.square(cp.norm2((np.array(Y[:,0]).reshape(L, )) - (cp.matmul(Phi[i], X))))) + cp.multiply(.001 ,cp.norm1(X)))
        obj = cp.Minimize(cp.multiply((1 / (2 * L)), cp.square(cp.norm2(Y_test[:, i] - (cp.matmul(Phi[i], X))))) + cp.multiply(.001, cp.norm1(X)))
        prob = cp.Problem(obj)
        prob.solve(solver = cp.CVXOPT) # ['CVXOPT', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI', 'OSQP', 'SCS']
        X_bar.append(X.value)
        # print("\n\n status: {} \n\n optimal value: {} \n\n optimal varible: x = {}".format(prob.status, prob.value, X.value))
    X_bar = np.matrix(X_bar) # not be exactly zero or one

    # test
    count_nl, count_el, count_fail = 0, 0, 0
    for i in range(X_bar.shape[0]):
        for j in range(X_bar.shape[1]):
            if X_bar[i, j] > -.1 and X_bar[i, j] < .1:
                count_nl += 1
                # X_bar[i, j] = 0
            elif X_bar[i, j] > .9 and X_bar[i, j] < 1.1:
                count_el += 1
                # X_bar[i, j] = 1
            else: count_fail += 1
    pre_fail.append(count_fail / (count_fail + count_el + count_nl))
    np.savetxt('A_M_bar.csv', X_bar, delimiter=',')
    end2 = time()

    A_M = np.loadtxt(open("C:\\Users\\27225\\Desktop\\workshop\\A_M.csv", "rb"), delimiter=",", skiprows=0)
    A_M_bar = np.loadtxt(open("C:\\Users\\27225\\Desktop\\workshop\\A_M_bar.csv", "rb"), delimiter=",", skiprows=0)

    # a = np.array((A_M.flatten(), A_M_bar.flatten()))
    # a = a.T[np.lexsort(-a)].T

    # compute the AUROC
    fpr, tpr, _ = roc_curve(A_M.flatten(), A_M_bar.flatten())
    roc_auc = auc(fpr, tpr)
    print("AUROC = {}".format(roc_auc))
    plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (AUROC = %0.2f)' % roc_auc)
    plt.legend(loc = 'best')
plt.show()

plt.figure()
plt.xlabel('MC Rounds')
plt.ylabel('Fail Rate')
plt.plot(range(50, 500, 50), pre_fail, lw = 2)
plt.show()

print("Total Running Time with {} Rounds MC and {} Rounds Fermi updating cost: {} s".format(L, G.number_of_nodes(), end1 - start1))
print("Running Time with Lasso solve cost: {} s".format(end2 - start2))