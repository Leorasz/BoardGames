import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

# torch.manual_seed(1337)

class Game():

    def __init__(self): self.judge = JudgerNet()

    def playMove(self, pos, player, prin=True):
        if self.state[pos-1] == 0:
            self.state[pos-1] = player+1
        else:
            print("INVALID MOVE, TRY AGAIN")
        if prin:
            self.showBoard()

    def checkWin(self):
        #vertical 
        for i in range(3):
            if self.state[i] == self.state[i+3] == self.state[i+6] != 0:
                # print("Vertical W")
                return self.state[i]

        #horizontal  
        for i in range(0,9,3):
            if self.state[i] == self.state[i+1] == self.state[i+2] != 0:
                # print("Horizontal W")
                return self.state[i]

        #diagonal
        if self.state[0] == self.state[4] == self.state[8] != 0:
            # print("TL Diagonal W")
            return self.state[0]
        if self.state[2] == self.state[4] == self.state[6] != 0:
            # print("TR Diagonal W")   
            return self.state[2]
        
        #check if full
        broken = False
        for i in self.state:
            if i == 0:
                broken = True
                break
        if not broken:
            #print("Full board L")
            return 3
        
        return 0
        
    def showBoard(self):
        for i in range(3):
            print(self.state[3*i:(3*i)+3])
        print("\n")
        #print(list(range(1,8)))

    def setBoard(self):
        self.state = [0]*9

    def humanvhuman(self):
        self.setBoard()
        self.showBoard()
        player2 = False
        while self.checkWin() == 0:
            self.playMove(int(input(f"Enter your position, player {int(player2)+1}:")), int(player2))
            player2 = not player2
        print(f"Player {int(not player2) + 1} wins!!!")

    def trainWithJudge(self, games, printmode=False):
        self.losses = []
        self.jlosses = []
        self.setBoard()
        self.showBoard()
        self.history = []
        self.actions = []

        self.bot = PlayerNet()
        self.judge.seq.state_dict(torch.load("tttjudgefor3.pt"))

        boptimizer = torch.optim.Adam(self.bot.seq.parameters(), lr=3e-4)
        bscheduler = torch.optim.lr_scheduler.StepLR(boptimizer, step_size=70000, gamma=0.1)
        joptimizer = torch.optim.Adam(self.judge.seq.parameters(), lr=3e-4)
        jscheduler = torch.optim.lr_scheduler.StepLR(joptimizer, step_size = 10000, gamma=0.1)
        for game in range(games):
            while True:
                # print("Now player 1's turn")
                action = self.bot.forward(self.player1state())
                self.actions.append(action)
                self.playMove(action+1, 0, prin=printmode)
                self.history.append(self.player1state())
                self.updateBotWithJudge(boptimizer)
                bscheduler.step()
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking1")
                        break
                    print("Player 1 win")
                    # self.updateJudge(joptimizer)
                    # jscheduler.step()
                    break
                # print("Now player 2's turn")
                action = self.bot.forward(self.player2state())
                self.actions.append(action)
                self.playMove(action+1, 1, prin=printmode)
                self.history.append(self.player2state())
                self.updateBotWithJudge(boptimizer)
                bscheduler.step()
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking2")
                        break
                    print("Player 2 win")
                    # self.updateJudge(joptimizer)
                    # jscheduler.step()
                    break
            print(f"Game {game+1}")
            self.history = []
            self.setBoard()
        if not printmode: 
            plt.plot(self.losses, color="red")
            plt.show()
            plt.clf
            plt.plot(self.jlosses, color="red")
            plt.show()
        torch.save(self.bot.seq.state_dict(), "judgedBot.pt")
        # torch.save(self.judge.seq.state_dict(), "tttjudgefor3.pt")

    def updateBotWithJudge(self, optimizer):
        hist = torch.tensor(self.history[-1], dtype=torch.float32).reshape(1,-1)
        outs = self.bot.batchForward(hist).reshape(1,-1)  # Ensure outs.requires_grad is True
        masked_outs = self.apply_mask_to_outputs(outs[0], self.history[-1])
        softmax = torch.nn.Softmax(dim=0)
        soft_outs = softmax(masked_outs)
        last_two_indices = self.actions[-1]  # e.g., [0, 3]
        selected_actions = soft_outs[last_two_indices]
        criterion = nn.MSELoss()
        reward = self.judge.forward(self.history[-1])
        loss = criterion(selected_actions, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.losses.append(loss.item())

    def updateJudge(self, optimizer, gamma=0.95):
        print("Updating judge")
        gammastore = 1
        criterion = nn.MSELoss()
        for board in self.history[::-1]:
            out = self.judge.forward(board)
            loss = criterion(out, torch.tensor([gammastore], dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gammastore *= -gamma
            self.jlosses.append(loss.item())

    def testJudge(self):
        self.judge.seq.load_state_dict(torch.load("tttjudgefor3.pt"))
        testcase1 = [1,0,-1,1,-1,0,1,0,0]
        testcase2 = [-1,0,1,1,-1,0,0,1,-1]
        testcase3 = [0]*9
        testcase4 = [1,0,1,-1,-1,1,-1,0,0]
        testcase5 = [-1,0,1,0,1,0,-1,0,-1]
        a = [testcase1, testcase2, testcase3, testcase4, testcase5]
        for tc in a:
            self.state = tc
            self.showBoard()
            print(self.judge.forward(tc).item())

    def apply_mask_to_outputs(self, outputs, board):
        # print(board)
        masked_outputs = outputs.clone()  # Clone to avoid modifying the original tensor
        for sindex, spot in enumerate(board):
            if spot != 0:
                masked_outputs[sindex] = float("-inf")  # Set to negative infinity to ignore this column
        return masked_outputs

    def humanvbot(self, loaded):
        self.setBoard()
        self.showBoard()
        while True:
            botmove = loaded.forward(self.player1state())
            self.playMove(botmove+1, 0)
            if self.checkWin() != 0:
                print("Bot wins!")
                break
            print("--------------------")
            self.playMove(int(input(f"Enter your position, human:")), 1)
            if self.checkWin() != 0:
                print("Human wins!")
                break
    
    def botBattle(self, games, bot1, bot2, printmode=True):
        self.setBoard()
        self.showBoard()
        bot1wins = 0
        bot2wins = 0
        for game in range(games):
            while True:
                # print("Now player 1's turn")
                action = bot1.forward(self.player1state())
                self.playMove(action+1, 0, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking1")
                        break
                    #print("Player 1 win")
                    bot1wins+=1
                    break
                # print("Now player 2's turn")
                action = bot2.forward(self.player2state())
                self.playMove(action+1, 1, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking2")
                        break
                    #print("Player 2 win")
                    bot2wins+=1
                    break
            #print(f"Game {game+1}")
            self.setBoard()
        for game in range(games):
            while True:
                # print("Now player 1's turn")
                action = bot2.forward(self.player1state())
                self.playMove(action+1, 0, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking1")
                        break
                    #print("Player 1 win")
                    bot2wins+=1
                    break
                # print("Now player 2's turn")
                action = bot1.forward(self.player2state())
                self.playMove(action+1, 1, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking2")
                        break
                    #print("Player 2 win")
                    bot1wins+=1
                    break
            #print(f"Game {game+1}")
            self.setBoard()
        
        print(f"Bot 1 won {bot1wins} times, bot 2 won {bot2wins} times")

    def trainWithoutJudge(self, games, printmode=False):
        self.losses = []
        self.setBoard()
        self.showBoard()
        self.history = []
        self.actions = []
        self.bot = PlayerNet()
        optimizer = torch.optim.Adam(self.bot.seq.parameters(), lr=3e-4)
        for game in range(games):
            while True:
                # print("Now player 1's turn")
                self.history.append(self.player1state())
                action = self.bot.forward(self.player1state())
                self.actions.append(action)
                self.playMove(action+1, 0, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking1")
                        break
                    print("Player 1 win")
                    self.updateBotWithoutJudge(optimizer)
                    break
                # print("Now player 2's turn")
                self.history.append(self.player2state())
                action = self.bot.forward(self.player2state())
                self.actions.append(action)
                self.playMove(action+1, 1, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking2")
                        break
                    print("Player 2 win")
                    self.updateBotWithoutJudge(optimizer)
                    break
            print(f"Game {game+1}")
            self.history = []
            self.actions = []
            self.setBoard()
        plt.plot(self.losses, color="red")
        if not printmode: plt.show()

    def updateBotWithoutJudge(self, optimizer):
        print("Going to update bot")
        hist = torch.tensor(self.history[-2:], dtype=torch.float32).reshape(2,-1)
        outs = self.bot.batchForward(hist).reshape(2,-1)  # Ensure outs.requires_grad is True
        masked_outs = torch.stack([self.apply_mask_to_outputs(outs[i], self.history[-2 + i]) for i in range(2)])
        softmax = torch.nn.Softmax(dim=1)
        soft_outs = softmax(masked_outs)
        last_two_indices = self.actions[-2:]  # e.g., [0, 3]
        selected_actions = soft_outs[torch.arange(soft_outs.size(0)), last_two_indices]
        criterion = nn.MSELoss()
        rewards = torch.tensor([-1, 1], dtype=torch.float32)
        loss = criterion(selected_actions, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss is {loss}")
        self.losses.append(loss.item())

    def trainWithOpponent(self, games, printmode=False):
        self.losses = []
        self.setBoard()
        self.showBoard()
        self.history = []
        self.actions = []
        self.bot = PlayerNet()
        destroyer1 = PlayerNet()
        destroyer1.seq.load_state_dict(torch.load("tttmodel2destroyer.pt"))
        destroyer2 = PlayerNet()
        destroyer2.seq.load_state_dict(torch.load("destroyerx2.pt"))
        model2 = PlayerNet()
        model2.seq.load_state_dict(torch.load("tttmodel2.pt"))
        model3 = PlayerNet()
        model3.seq.load_state_dict(torch.load("tttmodel3.pt"))
        opponents = [destroyer1, destroyer2, model2, model3]
        optimizer = torch.optim.Adam(self.bot.seq.parameters(), lr=3e-4)
        winrates = []
        wins = 0
        for game in range(games):
            opponent = opponents[games%len(opponents)]
            while True:
                # print("Now player 1's turn")
                self.history.append(self.player1state())
                action = self.bot.forward(self.player1state())
                self.actions.append(action)
                self.playMove(action+1, 0, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking1")
                        break
                    #print("Player 1 win")
                    wins+=1
                    if game % 100 == 1:
                        winrates.append(wins)
                        wins = 0
                    self.updateBotWithOpponent(optimizer, 0)
                    break
                # print("Now player 2's turn")
                self.history.append(self.player2state())
                action = opponent.forward(self.player2state())
                self.actions.append(action)
                self.playMove(action+1, 1, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking2")
                        break
                    #print("Player 2 win")
                    self.updateBotWithOpponent(optimizer, 1)
                    break
            # print(f"Game {game+1}")
            self.history = []
            self.actions = []
            self.setBoard()
            while True:
                # print("Now player 1's turn")
                self.history.append(self.player1state())
                action = opponent.forward(self.player1state())
                self.actions.append(action)
                self.playMove(action+1, 0, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking1")
                        break
                    #print("Player 1 win")
                    wins+=1
                    if game % 100 == 1:
                        winrates.append(wins)
                        wins = 0
                    self.updateBotWithOpponent(optimizer, 1)
                    break
                # print("Now player 2's turn")
                self.history.append(self.player2state())
                action = self.bot.forward(self.player2state())
                self.actions.append(action)
                self.playMove(action+1, 1, prin=printmode)
                if self.checkWin() != 0:
                    if self.checkWin == 3:
                        print("Breaking2")
                        break
                    #print("Player 2 win")
                    self.updateBotWithOpponent(optimizer, 0)
                    break
            if game % 1000 == 1:
                print(f"Game {game+1}")
            self.history = []
            self.actions = []
            self.setBoard()
        plt.plot(self.losses, color="red")
        if not printmode: 
            plt.show()
            plt.clf()
            plt.plot(winrates)
            plt.show()
        torch.save(self.bot.seq.state_dict(), "destroyerOfWorlds.pt")

    def updateBotWithOpponent(self, optimizer, winner):
        #print("Going to update bot")
        hist = torch.tensor(self.history[-1], dtype=torch.float32).reshape(1,-1) if winner == 0 else torch.tensor(self.history[-2], dtype=torch.float32).reshape(1,-1)
        outs = self.bot.batchForward(hist).reshape(1,-1)  # Ensure outs.requires_grad is True
        masked_outs = self.apply_mask_to_outputs(outs[0], self.history[-1])
        softmax = torch.nn.Softmax(dim=0)
        soft_outs = softmax(masked_outs)
        last_two_indices = self.actions[-1] if winner == 0 else self.actions[-2]
        selected_actions = soft_outs[last_two_indices]
        criterion = nn.MSELoss()
        rewards = torch.tensor([1], dtype=torch.float32) if winner == 0 else torch.tensor([-1], dtype=torch.float32)
        loss = criterion(selected_actions, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f"Loss is {loss}")
        self.losses.append(loss.item())

    def player1state(self):
        out = []
        for i in self.state:
            if i == 2:
                out.append(-1)
            else:
                out.append(i)
        return out

    def player2state(self):
        out = []
        for i in self.state:
            if i == 1:
                out.append(-1)
            elif i == 2:
                out.append(1)
            else:
                out.append(i)
        return out

class PlayerNet():
    def __init__(self):
        self.seq = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
            nn.Tanh()
        )

        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)  # It's also a good practice to initialize biases to zero

    def forward(self, x):
        xt = torch.tensor(x, dtype=torch.float32).reshape(1,-1)
        out = self.seq(xt).reshape(-1).tolist()
        out = torch.tensor([i if x[index]==0 else float("-inf") for index, i in enumerate(out)], dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=0)
        out = softmax(out)
        # print(out)
        distribution = torch.distributions.Categorical(out)
        return distribution.sample()

    def batchForward(self, x):
        return self.seq(x)

class JudgerNet():
    def __init__(self):
        self.seq = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, x):
        xt = torch.tensor(x, dtype=torch.float32).reshape(1,-1)
        out = self.seq(xt)
        return out
    
gam = Game()
# gam.trainWithOpponent(100_000)
destroyer1 = PlayerNet()
destroyer1.seq.load_state_dict(torch.load("tttmodel2destroyer.pt"))
destroyer2 = PlayerNet()
destroyer2.seq.load_state_dict(torch.load("destroyerx2.pt"))
model2 = PlayerNet()
model2.seq.load_state_dict(torch.load("tttmodel2.pt"))
model3 = PlayerNet()
model3.seq.load_state_dict(torch.load("tttmodel3.pt"))
opponents = [destroyer1, destroyer2, model2, model3]
destroyerOfWorlds = PlayerNet()
destroyerOfWorlds.seq.load_state_dict(torch.load("destroyerOfWorlds.pt"))
# for i in opponents: gam.botBattle(2000, destroyerOfWorlds, i, printmode=False)
gam.humanvbot(destroyerOfWorlds)
gam.humanvbot(destroyerOfWorlds)
gam.humanvbot(destroyerOfWorlds)