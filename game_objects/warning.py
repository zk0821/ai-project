import pygame

width, height = (854, 480)
screen = pygame.display.set_mode((width, height))

class TrainingWarning():

    def __init__(self):
        self.q_learning = False
        self.deep_q_learning = False
        self.font = pygame.font.Font("fonts/pixel_monospace_font.otf", 30)
        self.midi_font = pygame.font.Font("fonts/pixel_monospace_font.otf", 25)


    def draw(self):
        text_training = self.font.render("TRAINING MODE:" if self.q_learning or self.deep_q_learning else "", True, "red")
        screen.blit(text_training, (width * 0.3, height * 0.05))
        q_learning_training = self.midi_font.render("Q-Learning" if self.q_learning else "", True, "red")
        screen.blit(q_learning_training, (width * 0.37, height * 0.12))
        deep_q_learning_training = self.midi_font.render("Deep Q-Learning" if self.deep_q_learning else "", True, "red")
        screen.blit(deep_q_learning_training, (width * 0.31, height * 0.12))

    def update(self, q_learning, deep_q_learning):
        self.q_learning = q_learning
        self.deep_q_learning = deep_q_learning