import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

torch.manual_seed(1337)

class Game():

    def playMove(self, pos, player, prin=True):
        broken=False
        for layer in self.state:
            if layer[pos-1] == 0:
                layer[pos-1] = player + 1
                broken=True
                break
        if not broken:
            print("INVALID MOVE, TRY AGAIN")
        if prin:
            self.showBoard()

    def checkWin(self, dontWin=False):
        if dontWin:
            return 0
        # Check horizontal
        for row in self.state:
            for col in range(len(row) - 3):
                if row[col] == row[col+1] == row[col+2] == row[col+3] != 0:
                    print("Horizontal W")
                    return row[col]

        # Check vertical
        for col in range(len(self.state[0])):
            for row in range(len(self.state) - 3):
                if self.state[row][col] == self.state[row+1][col] == self.state[row+2][col] == self.state[row+3][col] != 0:
                    print("Vertical W")
                    return self.state[row][col]

        # Check diagonal (top-left to bottom-right)
        for row in range(len(self.state) - 3):
            for col in range(len(self.state[0]) - 3):
                if self.state[row][col] == self.state[row+1][col+1] == self.state[row+2][col+2] == self.state[row+3][col+3] != 0:
                    print("TL-BR Diagonal W")
                    return self.state[row][col]

        # Check diagonal (top-right to bottom-left)
        for row in range(len(self.state) - 3):
            for col in range(3, len(self.state[0])):
                if self.state[row][col] == self.state[row+1][col-1] == self.state[row+2][col-2] == self.state[row+3][col-3] != 0:
                    print("TR-BL Diagonal W")
                    return self.state[row][col]
        
        # Check to make sure the board isn't full
        full = True
        for spot in self.state[-1]:
            if spot == 0:
                full = False
                break
        if full:
            print("Full board L")
            return 3
            
        return 0  # No win detected
                     
    def showBoard(self):
        for i in self.state[::-1]:
            print(i)
        print("\n")
        #print(list(range(1,8)))

    def humanvhuman(self):
        self.state = [1,2,1,2,1,2,1]*2 + [[0 for _ in range(7)] for _ in range(4)]
        self.showBoard()
        player2 = False
        while self.checkWin() == 0:
            self.playMove(int(input(f"Enter your position, player {int(player2)+1}:")), int(player2))
            player2 = not player2
        print(f"Player {int(not player2) + 1} wins!!!")

    def botvbot(self, games, printmode=False):
        self.losses = []
        self.state = [[1,2,1,2,1,2,1]]*2 + [[0 for _ in range(7)] for _ in range(4)]
        self.showBoard()
        self.history = []
        self.actions = []
        self.bot = NeuralNet(0.2)
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
                    self.updateBot(optimizer)
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
                    self.updateBot(optimizer)
                    break
            print(f"Game {game+1}")
            self.history = []
            self.actions = []
            self.state = [[1,2,1,2,1,2,1]]*2 + [[0 for _ in range(7)] for _ in range(4)]
        plt.plot(self.losses, color="red")
        #if not printmode:
        plt.show()

    def updateBot(self, optimizer):
        print("Going to update bot")
        hist = torch.tensor(self.history[-2:], dtype=torch.float32).reshape(2,-1)
        # print(hist[0].reshape(6,7))
        # print(hist[1].reshape(6,7))
        # print(torch.tensor(self.history[-1]).reshape(6,7))
        # print(hist)
        outs = self.bot.batchForward(hist).reshape(2,-1)  # Ensure outs.requires_grad is True
        # print(outs)
        # Apply mask to the outputs for both boards
        masked_outs = torch.stack([self.apply_mask_to_outputs(outs[i], self.history[-3 + i]) for i in range(2)])
        # print(masked_outs)
        # Using softmax with low temperature to approximate argmax
        # print(masked_outs)
        temperature = 0.01
        softmax = torch.nn.Softmax(dim=1)
        soft_outs = softmax(masked_outs)
        # print(soft_outs)
        # Selecting the maximum action using softmax
        last_two_indices = self.actions[-2:]  # e.g., [0, 3]
        selected_actions = soft_outs[torch.arange(soft_outs.size(0)), last_two_indices]
        # print(selected_actions)
        criterion = nn.MSELoss()
        rewards = torch.tensor([-1, 1], dtype=torch.float32)
        loss = criterion(selected_actions, rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss is {loss}")
        self.losses.append(loss.item())

    def apply_mask_to_outputs(self, outputs, board):
        masked_outputs = outputs.clone()  # Clone to avoid modifying the original tensor
        for col in range(7):
            if self.is_column_full(board, col):
                masked_outputs[col] = float("-inf")  # Set to negative infinity to ignore this column
        return masked_outputs

    def is_column_full(self, board, col):
        # Assuming 0 is the empty cell
        # Check if the top cell of the column is not empty
        return board[35+col] != 0

    def humanvbot(self):
        self.state = [[1,2,1,2,1,2,1]]*2 + [[0 for _ in range(7)] for _ in range(4)]
        self.showBoard()
        while True:
            self.playMove(int(input(f"Enter your position, human:")), 0)
            if self.checkWin() != 0:
                print("Human wins!")
                break
            print("--------------------")
            botmove = self.bot.forward(self.player2state())
            self.playMove(botmove+1, 1)
            if self.checkWin() != 0:
                print("Bot wins!")
                break


    def player1state(self):
        out = []
        for row in self.state:
            for i in row:
                if i == 2:
                    out.append(-1)
                else:
                    out.append(i)
        return out

    def player2state(self):
        out = []
        for row in self.state:
            for i in row:
                if i == 1:
                    out.append(-1)
                elif i == 2:
                    out.append(1)
                else:
                    out.append(i)
        return out

class NeuralNet():
    def __init__(self, dropout_rate):
        self.seq = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),  # Add dropout after ReLU
            nn.Linear(32, 7),
            nn.Tanh()
        )

        for layer in self.seq:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)  # It's also a good practice to initialize biases to zero

    def forward(self, x):
        # print(torch.tensor(x).reshape(6,7))
        xt = torch.tensor(x, dtype=torch.float32).reshape(1,-1)
        out = self.seq(xt).reshape(-1).tolist()
        out = torch.tensor([i if x[35+index] == 0 else float("-inf") for index, i in enumerate(out)], dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=0)
        out = softmax(out)
        # print(out)
        distribution = torch.distributions.Categorical(out)
        return distribution.sample()

    def batchForward(self, x):
        return self.seq(x)
    
gam = Game()
gam.botvbot(10000, printmode=True)
gam.humanvbot()
# gam.humanvhuman()

    