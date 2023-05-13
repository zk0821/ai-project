import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Linear_Q_Neural_Network(nn.Module):


    def __init__(self, input_size, hidden_size, output_size):
        """Initializing the linear neural network

        Args:
            input_size (_type_): size of input vector
            hidden_size (_type_): size of neural network hidden layers
            output_size (_type_): size of output vector
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.load_model()


    def forward(self, x) :
        """Takes input vector and passes it through the Neural Network applying relu activation function and returning the output. In other words this is the prediction function that will be called by the agent

        Args:
            x (vector): state vector

        Returns:
            vector: next move decided by the neural network
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


    def save_model(self, filename="linear_deep_q_learning_neural_network.pth"):
        """Saves the model for future use

        Args:
            filename (str, optional): name of the file where the model will be saved. Defaults to "linear_deep_q_learning_neural_network.pth".
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename="linear_deep_q_learning_neural_network_lots_of_epochs.pth"):
        """Loads the model

        Args:
            filename (str, optional): name of the file where the model is saved. Defaults to "linear_deep_q_learning_neural_network.pth".
        """
        torch.load(filename)
        torch.eval()


class QTrainer:


    def __init__(self, learning_rate, gamma, model):
        """Initialization function for the QTrainer class

        Args:
            learning_rate (float): Learning rate for the Optimizer
            gamma (_type_): Discount rate used in the Bellman equation
            model (_type_): Linear_Q_Neural_Network
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        # Optimizer for weight and bias updates
        self.optimizer = optim.Adam(model.parameters(), weight_decay=0, lr = self.learning_rate)


    def train_single_step(self, _state, _next_state, _action, _reward, done):
        """Training the model

        Args:
            _state (_type_): Current state
            _next_state (_type_): Next state
            _action (_type_): Action taken
            _reward (_type_): Reward for state
            done (function): Is runner dead
        """
        self.model.train()
        torch.set_grad_enabled(True)
        target = _reward
        next_state = np.asarray(_next_state)
        next_state_tensor = torch.tensor(next_state.reshape((1, 4)), dtype=torch.float32)
        state = np.asarray(_state)
        state_tensor = torch.tensor(state.reshape((1, 4)), dtype=torch.float32, requires_grad=True)
        if not done:
            target = _reward + self.gamma * torch.max(self.model.forward(next_state_tensor[0]))
        output = self.model.forward(state_tensor)
        target_f = output.clone()
        target_f[0][_action] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()


    def train_multi_step(self, mini_sample):
        for _state, _next_state, _action, _reward, _done in mini_sample:
            self.train_single_step(_state, _next_state, _action, _reward, _done)