import json
import random

actions = {
    'IDLE': 0,
    'UP': 1,
    'DOWN': 2,
}

class Bot(object):
    def __init__(self):
        self.games_played = 0
        self.discount = 0.05
        self.reward = {0: 1000, 1: -1000}
        self.alpha = 0.7
        self.last_state = None
        self.last_action = 0
        self.q_values = self.load_q_values()
        self.actions = [actions['IDLE'], actions['UP'], actions['DOWN'],]
        self.init_state = [0] * (len(self.actions))
        self.exploration_prob = 80
        self.action_cooldown = 0
        self.scores_per_game=[]
        self.rewarding_enabled = True

    def get_exploration_val(self):
        return self.exploration_prob / (self.games_played + 1)

    def load_q_values(self):
        try:
            q_value_file = open('qvalues.json', 'r')
            q_values = json.load(q_value_file)
            q_value_file.close()
            return q_values
        except (IOError, ValueError):
            print("Error while loading qvalues.json")
            return {}

    def dump_q_values(self):
        fil = open('qvalues.json', 'w')
        json.dump(self.q_values, fil)
        fil.close()
        print('Q-values updated on local file.')

    def act(self, x_diff, obstacle_y, game_speed, obstacle=1):
        self.action_cooldown -= 1
        distance_threshold = (100 + game_speed) if (obstacle == 1 and obstacle_y < 300) else (110 + game_speed*3)

        if x_diff > distance_threshold or self.action_cooldown > 0:
            if not self.last_state:
                state = self.map_state(x_diff, obstacle_y, game_speed, obstacle)
                self.last_state = state
                self.last_action = 0
            else: return 0
        else:
            state = self.map_state(x_diff, obstacle_y, game_speed, obstacle)
            self.last_state = state
            if state not in self.q_values:
                self.q_values[state] = [0] * (len(self.actions))
                self.last_action = 0
            else:
                best_action = self.get_best_action(state)
                self.last_action = best_action
        return self.last_action

    def update_q_values(self, game_param, is_dead=False, user_event=None):

        if self.last_state not in self.q_values:
            self.q_values[self.last_state] = [0] * (len(self.actions))

        if is_dead:
            self.games_played += 1
            print(f"KAZNUJEMO akcijo: {self.last_action} v stanju {self.last_state}")

            self.q_values[self.last_state][self.last_action] = (1 - self.alpha) * (self.q_values[self.last_state][self.last_action]) + self.alpha * (
                self.reward[1] + self.discount * max(self.q_values[self.last_state]))

            if self.games_played % 10 == 0:  self.dump_q_values()

        if not is_dead and self.rewarding_enabled:
            print(f"NAGRAJUJEMO akcijo: {self.last_action} v stanju {self.last_state}")
            self.q_values[self.last_state][self.last_action] = (1 - self.alpha) * (self.q_values[self.last_state][self.last_action]) + self.alpha * (
                self.reward[0] + self.discount * max(self.q_values[self.last_state]))


    def map_state(self, x_diff, y_obstacle, game_speed, obstacle=1):
        x_diff_step = 5
        x_diff_discrete = int(x_diff // x_diff_step) * x_diff_step
        return "{}_{}_{}_{}".format(x_diff_discrete, y_obstacle, game_speed, obstacle)

    def get_best_action(self, state):

        exp_val = self.get_exploration_val()
        epsilon = random.randint(0, 100) < exp_val #Not so much needed anymore

        if epsilon:
            action = random.choice([0, 1, 2])
        else:
            # Choose a random action out of the action with highest q values for this state
            action_q_values = [self.q_values[state][action] for action in [0, 1, 2]]
            max_indices = [index for index, value in enumerate(action_q_values) if value == max(action_q_values)]
            action = random.choice(max_indices)
            print(f"Chosen action {action} out of {action_q_values} for state {state}")
        return action