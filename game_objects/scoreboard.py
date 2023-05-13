import pygame
from helper_functions import load_image

width, height = (854, 480)
screen = pygame.display.set_mode((width, height))

margin = 0.75

class Scoreboard():

    def __init__(self, x=-1, y=-1):
        self.score = 0
        self.high_score = 0
        self.large_font = pygame.font.Font("fonts/pixel_monospace_font.otf", 18)
        self.small_font = pygame.font.Font("fonts/pixel_monospace_font.otf", 14)
        self.scoreboard_image, self.scoreboard_rect = load_image('scoreboard.png', 180, 130, -1)
        self.scoreboard_rect.left = width * margin
        self.scoreboard_rect.top = height * 0.05


    def draw(self):
        color = "black"
        anti_aliasing = True
        # Draw the scoreboard first
        screen.blit(self.scoreboard_image, self.scoreboard_rect)
        # Scoreboard
        text_scoreboard = self.large_font.render("SCOREBOARD", anti_aliasing, color)
        screen.blit(text_scoreboard, (width * (margin + 0.02), height * 0.07))
        # Separator
        text_line = self.large_font.render("------------", anti_aliasing, color)
        screen.blit(text_line, (width * margin, height * 0.095))
        # High score
        text_high_score = self.small_font.render("HIGHSCORE", anti_aliasing, color)
        screen.blit(text_high_score, (width * (margin + 0.02), height * 0.13))
        text_high_score_value = self.small_font.render(self.high_score, anti_aliasing, color)
        screen.blit(text_high_score_value, (width * (margin + 0.02), height * 0.17))
        # Current score
        text_score = self.small_font.render("SCORE", anti_aliasing, color)
        screen.blit(text_score, (width * (margin + 0.02), height * 0.21))
        text_score_value = self.small_font.render(self.score, anti_aliasing, color)
        screen.blit(text_score_value, (width * (margin + 0.02), height * 0.25))


    def update(self, score, high_score):
        self.score = str(score).zfill(12)
        self.high_score = str(high_score).zfill(12)