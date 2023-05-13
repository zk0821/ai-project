import pygame
from helper_functions import load_image

width, height= (854, 480)
screen = pygame.display.set_mode((width, height))

class Ground():
    def __init__(self, speed=-5):
        self.initialize_images()
        self.set_initial_position(height=height)
        self.speed = speed

    def initialize_images(self):
        self.image, self.rect = load_image('desert_background.png', 854, 480, 1)
        self.image1, self.rect1 = load_image('desert_background.png', 854, 480, 1)

    def set_initial_position(self, height):
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right

    def draw(self):
        screen.blit(self.image, self.rect)
        screen.blit(self.image1, self.rect1)

    def update(self):
        self.rect.left  = self.rect.left + self.speed
        self.rect1.left = self.rect1.left + self.speed

        self.rect.left = self.rect1.right if self.rect.right < 0 else self.rect.left
        self.rect1.left = self.rect.right if self.rect1.right < 0 else self.rect1.left