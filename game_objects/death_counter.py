import pygame
from helper_functions import load_image

width, height = (854, 480)
screen = pygame.display.set_mode((width, height))

class DeathCounter():
    def __init__(self):
        self.death_count = 0
        self.font = pygame.font.Font("fonts/pixel_monospace_font.otf", 20)
        self.death_image, self.death_rect = load_image('death_skull.jpg', 50, 50, -1)
        self.death_rect.left = width * 0.04
        self.death_rect.top = height * 0.055
        self.scoreboard_image, self.scoreboard_rect = load_image('scoreboard.png', 120, 60, -1)
        self.scoreboard_rect.left = width * 0.04
        self.scoreboard_rect.top = height * 0.05


    def draw(self):
        color = 'black'
        anti_aliasing = True
        # Draw the scoreboard first
        screen.blit(self.scoreboard_image, self.scoreboard_rect)
        # Draw the skull
        screen.blit(self.death_image, self.death_rect)
        # Death counter
        text_death_count = self.font.render("x" + str(self.death_count).zfill(3), anti_aliasing, color)
        screen.blit(text_death_count, (width * 0.095, height * 0.085))


    def update(self, death_count):
        self.death_count = death_count