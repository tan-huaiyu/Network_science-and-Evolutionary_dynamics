# Huaiyu TAN 2022-1-20

# N: is the number of Network.
# L: is the number of dynamics rounds.
# net_type: has three options:("BA"、"ER"、"SW"), which parameters inside this three can be changed build-in.
# game_type: has three options:("PDG"、"SDG"、"DG"), which parameters inside this three can be changed build-in.

#一共包含三种网络结构（BA无标度、ER随机、SW小世界）、四种博弈框架（囚徒困境博弈PDG、雪堆博弈SDG、捐赠博弈DG、SUG结构最后通牒博弈）

gen_dynamic<-function(N, L, net_type="BA", game_type="PDG"){

  ifelse(library(igraph) == FALSE, install.packages("igraph"), library(igraph)) # 基于igraph包生成网络
  # ifelse(library(bigmemory) == FALSE, install.packages("bigmemory"), library(bigmemory))
  
  # Calculate the payoff between i and j
  # which need Payoff Matrix（for others except SUG）
  payoff<-function(si, sj, M){
    return(t(as.matrix(si)) %*% M %*% as.matrix(sj))
  }
  # which no need Payoff Matrix（for SUG）
  payoff_sug<-function(s_i, s_j, S){
    p_i<-S[,s_i][1]; q_j<-S[,s_j][2]; p_j<-S[,s_j][1]; q_i<-S[,s_i][2]
    if ((p_i >= q_j) & (p_j >= q_i)) {p_sug<-p_j + 1 - p_i}
    if ((p_i >= q_j) & (p_j < q_i)) {p_sug<-1 - p_i}
    if ((p_i < q_j) & (p_j >= q_i)) {p_sug<-p_j}
    if ((p_i < q_j) & (p_j < q_i)) {p_sug<-0}
    return(p_sug)
  }
  
  # Generate different Game structure except SUG
  gen_PDG<-function(m.R, m.S, m.T, m.P){
    return(matrix(c(m.R, m.S, m.T, m.P), 2, 2, byrow=T)) # PDG Payoff Matrix
  }
  gen_SDG<-function(m.r){
    return(matrix(c(1, 1-m.r, 1+m.r, 0), 2, 2, byrow=T)) # SDG Payoff Matrix
  }
  gen_DG<-function(m.b, m.c){
    return(matrix(c(m.b-m.c, -m.c, m.b, 0), 2, 2, byrow=T)) # DG Payoff Matrix
  }
  
  ################################### Simulation of Dynamics ############################################
  
  # Generate different Complex Networks
  if(net_type == "BA"){G<-barabasi.game(n=N, power=1, m=6, directed = F)} # BA Scale-Free
  if(net_type == "ER"){G<-erdos.renyi.game(n=N, p.or.m=0.1, type="gnp", directed = F)} # ER Random
  if(net_type == "SW"){G<-sample_smallworld(dim=1, size=N, nei=3, p=0.01, directed = F)} # SW Regular
  
  # Init Payoff Matrix
  if(game_type == "PDG"){M<-gen_PDG(1, 0, 1.2, 0)} # R、S、T、P参数随意设置
  if(game_type == "SDG"){M<-gen_SDG(1.2)} # r参数随意设置
  if(game_type == "DG"){M<-gen_DG(2, 1)} # b、c参数随意设置
  
  # Init Strategy
  if(game_type == "SUG"){
	# for SUG
    S<-as.data.frame(matrix(NA, 2, N))
    for (i in 1:N) {
      S[,i]<-c(runif(1), runif(1))
    }
  }else{
    # For others
    S<-as.data.frame(matrix(NA, 2, N))
    for (i in 1:N) {
      if(runif(1) >= .5){
        S[,i]<-c(0, 1)
      }else{
        S[,i]<-c(1, 0)
      }
    }
  }
  
  # Record Adj
  real_adj<-as.matrix(get.adjacency(G))
  storage.mode(real_adj)<-"integer" # 因为Adj全是0-1，所以强行由double型储存为integer型
  
  # init X, Y
  Y<-data.frame(matrix(NA, L, N)) # 每一列为一个时间步的累计收益
  X<-list() # 为包含N个矩阵的列表（即分块对角矩阵分块存储）
  for (i in 1:N) {
    X[[i]]<-matrix(NA, L, N)
  }
  
  # simulation process
  for (l in 1:L) {
    F_temp<-matrix(NA, N, N)
    if(game_type == "SUG"){
      for (u in 1:N) {
        for (v in 1:N) {
          F_temp[u, v]<-payoff_sug(u, v, S)
        }
      }
    }else{
      for (u in 1:N) {
        for (v in 1:N) {
          F_temp[u, v]<-payoff(S[,u], S[,v], M)
        }
      }
    }
    
    # Record Y and X
    F_cumulate<-apply((real_adj*F_temp), 1, sum)
    Y[l, ]<-F_cumulate # The row of Y
    for (n in 1:N) {
      X[[n]][l,]<-F_temp[n,]
    }
    
    # Fermi Updating (Synchronization Update)
    if(game_type == "SUG"){
      for (i in 1:N) {
        update_index<-sample(as.vector(neighbors(G, i, "all")), 1)
        w<-1/(1 + exp((F_cumulate[i] - F_cumulate[update_index])/.1))
        if(runif(1) >= w){
          S[, i]<-S[, i] + .05
        }
      }
    }else{
      for (i in 1:N) {
        update_index<-sample(as.vector(neighbors(G, i, "all")), 1)
        w<-1/(1 + exp((F_cumulate[i] - F_cumulate[update_index])/.1))
        if(runif(1) >= w){
          S[, i]<-S[, update_index]
        }
      }
    }
  }
  # X block diag treatment
  # X_diag_block<-as.matrix(bdiag(X)) #使用此代码将X保存为分块对角矩阵需要用到开头备注的 “bigmemory” 包
  return(list(Y = Y, X = X, real_adj = real_adj))
}

# 使用方法举个例子
# dyn_vars<-gen_dynamic(100, 15, "BA", "PDG")
# X<-dyn_vars[["X"]]
# Y<-dyn_vars[["Y"]]
# X<-X[[1]]
# Y<-Y[,1]

# dyn_vars中包含Y、X、Adj
# 调用方法：dyn_vars[["Y"]]、dyn_vars[["X"]]、dyn_vars[["real_adj"]]
# X又分dyn_vars[["X"]][[1]]、dyn_vars[["X"]][[2]]、.... 、dyn_vars[["X"]][[N]]