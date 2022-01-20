import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Define Useful Tools
class Use_Tools:
    def payoff(si, M, sj) -> float:  # si*M*sj' and input is np.array
        phi = np.matmul(np.matmul(si, M), sj.T)
        return phi
    def findneb(p, node_list, edge_list): # 游戏中根据网络选择机制找邻居
        focal_player = node_list[p] - 1  # n+1 number node
        rdn_neb = []  # n+1 number node
        for i in range(len(edge_list)):
            if edge_list[i][0] != focal_player:
                if edge_list[i][1] == focal_player:
                    rdn_neb.append(edge_list[i][0])
            else:
                rdn_neb.append(edge_list[i][1])
        return rdn_neb, focal_player
    def gen_A_fake(n: int, p: float): # 生成伪邻接矩阵
        A = np.triu(np.random.binomial(1, p, size=(n, n)))
        A += A.T - np.diag(A.diagonal())
        i = range(n)
        A[i, i] = 0 # 对角线元素为0
        return A
    def auroc_mat(a_real, a_fake): # 用于检验真邻接矩阵与重构矩阵的AUROC
        fpr, tpr, thersholds = roc_curve(a_fake.flatten(), a_real.flatten())
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc

# Define Game Structure
class Game:
    def PDG(b: float):
        M = np.array([[1, 0], [b, 0]])  # b = 1.2; M[0][0]=1; M[0][1]=0; M[1][0]=1.2; M[1][1]=0
        return M
    def SDG(r: float):
        M = np.array([[1, 1 - r], [1 + r, 0]])
        return M

# Define net Structure
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

N = 100; k = 3
G = Gen_Net.basf(N, k)  # average degree <k> = 6: k = 3
node_list = np.array(list(G.nodes)) + 1  # all nodes
nebmat = nx.to_numpy_matrix(G)  # Adjacent Matrix of G
np.savetxt('A_Real.csv', nebmat, delimiter=',')

class Gen_Real():
    def __init__(self, L):
        self.L = L
    def MC_Fermi(self): # Play MC (synchronous updating with Fermi)
        # Initalize Strategies Set
        s_set = []; s = []  # every node which choose C or D with equal prob
        for _ in range(N):
            temp_rand = random.random()
            if temp_rand >= .50:
                s_set.append((0, 1))
            else:
                s_set.append((1, 0))  # strategies set of #N nodes
        # D_init = 0;
        # C_init = 0
        # for i in range(N):
        #     if (self.s_set[i] == (0, 1)):
        #         D_init += 1
        #     else:
        #         C_init += 1
        # coop_rate_init = C_init / (C_init + D_init)  # init coopration rate
        edge_list = np.array(G.edges); M = Game.PDG(1.2); trangle_mat_index = nebmat[np.triu_indices(N)]
        f_temp = np.zeros([N, N])
        f = np.zeros([N, self.L])
        for l in range(self.L):
            # print("Round: {}: {}".format(l, s_set))
            s.append(s_set.copy())
            for i in range(N):
                for j in range(N):
                    f_temp[i][j] = Use_Tools.payoff(np.array(s_set[i]), M, np.array(s_set[j]))
            f_vec = np.sum(np.array(nebmat) * f_temp.T, 0)
            for n in range(N):
                (rdn_neb, focal_player) = Use_Tools.findneb(n, node_list, edge_list)
                select_neb = random.sample(rdn_neb, 1)[0]
                fi_lambda = np.sum(nebmat[focal_player])  # normlization
                fj_lambda = np.sum(nebmat[select_neb])
                w = 1 / (1 + np.exp((f_vec[focal_player]/fi_lambda - f_vec[select_neb]/fj_lambda) / 0.1))  # k = .1
                if random.random() >= w: s_set[focal_player] = s_set[select_neb]
            f[:, l] = f_vec.reshape(N, )
            # print("Round {}: {}".format(l + 1, s_set))
        return f, s
if __name__ == '__main__':
    (f, s) = Gen_Real(50).MC_Fermi()
    A_fake = Use_Tools.gen_A_fake(N, .15)
    np.savetxt('A_M_fake.csv', A_fake, delimiter=',')
    print("\n\n {} \n\n {} \n\n {}".format(f, s, A_fake))

    graph = nx.from_numpy_matrix(A_fake)
    plt.subplot(121)
    nx.draw(G)
    plt.title("Real")
    plt.subplot(122)
    nx.draw(graph)
    plt.title("Fake")
    plt.show()

    fpr, tpr, thersholds = roc_curve(A_fake.flatten(), np.array(nebmat).flatten())
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, 'k--', label='ROC (AUC = {0:.4f})'.format(roc_auc), lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc = "lower right")
    plt.show()