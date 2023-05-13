import torch
import random
import numpy as np
from collections import deque
from deep_q_learning.model import Linear_Q_Neural_Network, QTrainer

MAX_MEMORY = 10_000
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

class Agent:


    def __init__(self):
        """Initialization function for Agent
        """
        self.n_game = 0
        # Chance to take a random action
        self.epsilon = 0
        # Discount rate
        self.gamma = 0.05
        # Memory
        self.memory = deque(maxlen=MAX_MEMORY)
        # Model
        self.model = Linear_Q_Neural_Network(4, 256, 3)
        # Trainer
        self.trainer = QTrainer(learning_rate=LEARNING_RATE, gamma=self.gamma, model=self.model)
        # Current state
        self.current_state = []
        # Next state
        self.next_state = []
        # Chosen action
        self.chosen_action = 0
        # Action cooldown
        self.action_cooldown = 0


    def get_state(self, state):
        """Function to return the current state based on the game

        Args:
            state (vector): game state

        Returns:
            np.array: state
        """
        x_discretized = int(state[0] // 5) * 5
        return np.array([x_discretized, state[1], state[2], state[3]], dtype=float)


    def get_action(self):
        """Function to return the next move

        Args:
            state (vector): Current state of the game

        Returns:
            int: The next move to be made
        """
        # Random move: tradeoff between exploration and exploitation
        self.epsilon: int = 80 - self.n_game
        # self.epsilon = 20
        final_move = 0
        if (random.randint(0, 200) < self.epsilon):
            # Take a random move
            print("[Deep Q-Learning] Random action taken!")
            move = random.randint(0, 2)
            final_move = move
        else:
            # Predict the next move
            print("[Deep Q-Learning] Chosen action taken!")
            current_state = torch.tensor(self.current_state, dtype=torch.float)
            prediction = self.model(current_state)
            print(prediction)
            move = torch.argmax(prediction).item()
            final_move = move
        self.chosen_action = final_move
        return final_move


    def train_short_memory(self, reward, done):
        """Short memory training

        Args:
            reward (_type_): Reward for state
            done (function): Is the runner dead
        """
        self.trainer.train_single_step(self.current_state, self.current_state, self.chosen_action, reward, done)


    def train_long_memory(self):
        """Long term memory training
        """
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        self.trainer.train_multi_step(mini_sample)


    def remember(self, reward, done):
        """Short memory training

        Args:
            reward (_type_): Reward for state
            done (function): Is the runner dead
        """
        self.memory.append((self.current_state, self.current_state, self.chosen_action, reward, done))


    def decide_action(self):
        # Decrement the action cooldown
        self.action_cooldown -= 1
        # Only make action if within threshold
        distance_threshold = (80 + self.current_state[3]) if (self.current_state[2] == 1 and self.current_state[1] < 300) else (90 + self.current_state[3]*3)
        # Make decision
        decision = 0
        if self.current_state[0] < distance_threshold and self.action_cooldown <= 0:
            decision = self.get_action()
            self.chosen_action = decision
        else:
            decision = 0
        # Return the decision
        return decision