import pygame
from pygame import *
from helper_functions import load_image
import random

width, height = (854, 480)
screen = pygame.display.set_mode((width, height))


class Cactus(pygame.sprite.Sprite):
    def __init__(self, speed=5):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.initialize_images()
        self.set_initial_position(height=height)
        self.set_velocity(speed=speed)

    def initialize_images(self):
        self.image, self.rect = load_image('cactus.png', 100, 100, -1)

    def set_initial_position(self, height):
        self.rect.bottom = int(0.90 * height)
        self.rect.left = width + self.rect.width + random.randrange(0,100)

    def set_velocity(self, speed):
        self.movement = [-1 * speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right <= -1:
            self.kill()
