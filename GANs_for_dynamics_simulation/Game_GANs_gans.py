import torch, time
import numpy as np
import Game_GANs_game
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

L = 500 # MC Rounds
M = Game_GANs_game.Game.PDG(1.2)
(f_real, s) = Game_GANs_game.Gen_Real(L).MC_Fermi()
A_fake_init = Game_GANs_game.Use_Tools.gen_A_fake(Game_GANs_game.N, .15) # target

# G_NN
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        x = F.elu(self.map1(x))
        #x = F.logsigmoid(self.map2(x))
        x = torch.sigmoid(self.map2(x))
        return self.map3(x)

# D_NN
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        x = F.relu(self.map1(x))
        x = F.elu(self.map2(x))
        return torch.sigmoid(self.map3(x))

start = time.time()
# parameters option
g_input_size = Game_GANs_game.N; g_hidden_size = 25; g_out_size = Game_GANs_game.N
d_input_size = Game_GANs_game.N; d_hidden_size = 25; d_out_size = 1
d_learning_rate = 1e-2; g_learning_rate = 1e-2
optim_betas = (.9, .999); num_epochs = L; print_interval = 1
d_steps = 1; g_steps = 1

D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, out_size=d_out_size)
G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, out_size=g_out_size)

criterion = nn.BCELoss()

d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

# train D_NN and G_NN
d_loss_list = []; g_loss_list = []; dg_f_d_list = []
for epoch in range(num_epochs):
    for d_index in range(d_steps):  # steps in the original GAN paper. Can put the discriminator on higher training freq than generator
        # train D on real
        D.zero_grad()
        d_real_data = torch.from_numpy(f_real[:, epoch]); d_real_data = d_real_data.to(torch.float32)
        d_real_decision = D(d_real_data)
        d_real_error = criterion(d_real_decision, torch.ones(1, 1)[0]) # 记真为1，则将真实数据投入D并让D打分，越接近1误差越小，从而更新D
        d_real_error.backward()
        # d_real_error = criterion_D_real(d_real_decision)
        # # train D on fake
        f_temp = np.zeros([Game_GANs_game.N, Game_GANs_game.N])
        for i in range(Game_GANs_game.N):
            for j in range(Game_GANs_game.N):
                f_temp[i][j] = Game_GANs_game.Use_Tools.payoff(np.array(s[epoch][i]), M, np.array(s[epoch][j]))
        f_fake_sample = np.sum(A_fake_init * f_temp.T, 0)  # connect to target
        d_gen_input = torch.from_numpy(f_fake_sample); d_gen_input = d_gen_input.to(torch.float32)
        d_fake_data = G(d_gen_input).detach() # detach to avoid training G on these labels
        d_fake_decision = D(d_fake_data)
        d_fake_error = criterion(d_fake_decision, torch.zeros(1, 1)[0]) # 记假为0，则将假数据投入D并让D打分，越接近0误差越小，从而从另一方面更新D
        # d_fake_error = criterion_D_fake(d_fake_decision)
        # d_loss = (d_real_error + d_fake_error) / 2
        d_fake_error.backward()
        d_optimizer.step()

    for g_index in range(g_steps):
        # train G on D's response
        G.zero_grad()
        gen_input = torch.from_numpy(f_real[:, epoch]); gen_input = gen_input.to(torch.float32)
        g_fake_data = G(gen_input)
        dg_fake_decision = D(g_fake_data)
        g_loss = criterion(dg_fake_decision, torch.ones(1, 1)[0])
        # g_loss = criterion_G(dg_fake_decision)  # 再次把假数据投入已经训练完一轮的D并让D打分，并与1比较误差，D评分越接近于1，说明G的造假能力越强
        g_loss.backward()
        g_optimizer.step()

    # training A
    # loss_A = ||G(F_fake) - (Phi * A_fake)||_2^2 ——> 0
    # 遗传算法反复迭代求解A_fake

    plt.figure(figsize=(8, 6), dpi = 100)
    plt.xlabel('Node Number')
    plt.ylabel('Total Payoff')
    plt.title('Epoch {}'.format(epoch + 1))
    plt.plot(range(0, Game_GANs_game.N, print_interval), d_real_data, lw=2, color='black', label = "Real Payoff")
    plt.plot(range(0, Game_GANs_game.N, print_interval), d_fake_data, lw=2, color='red', label = "GANs Learning data")
    plt.legend(loc = "best")
    plt.savefig("C:/Users/killspeeder/Desktop/workshop/gifpic/pic{}".format(epoch))
    plt.clf()
    if epoch % print_interval == 0:
        d_loss_list.append(d_fake_error.item())
        g_loss_list.append(g_loss.item())
        dg_f_d_list.append(dg_fake_decision.item())
        print("{} \t {}".format(d_real_data, d_fake_data))
        print("Epoch {} ===> D_loss={}      G_loss={}      D_fake_dec={}".format(epoch + 1, d_fake_error.item(), g_loss.item(), dg_fake_decision.item()))



end = time.time()

plt.figure()
plt.xlabel('Epoch of GANs')
plt.ylabel('Generator Loss')
plt.plot(range(0, num_epochs, print_interval), g_loss_list, lw=2)
plt.show()

plt.figure()
plt.xlabel('Epoch of GANs')
plt.ylabel('D Loss')
plt.plot(range(0, num_epochs, print_interval), d_loss_list, lw=2, color='darkorange')
plt.show()

plt.figure()
plt.xlabel('Epoch of GANs')
plt.ylabel('D Fake to Real Score')
plt.plot(range(0, num_epochs, print_interval), dg_f_d_list, lw=2, color='black')
plt.show()

print("total Epoch of GANs time: {}".format(end - start))