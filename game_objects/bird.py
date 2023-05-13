import random
import pygame
from pygame import *
from helper_functions import load_sprite_sheet

width, height = (854, 480)
screen = pygame.display.set_mode((width, height))

class Bird(pygame.sprite.Sprite):
    def __init__(self, speed=5):
        pygame.sprite.Sprite.__init__(self, self.containers)
        self.initialize_images()
        self.set_initial_position()
        self.set_velocity(speed)
        self.index = 0
        self.counter = 0

    def initialize_images(self):
        self.images, self.rect = load_sprite_sheet('bird.png', 6, 2, 60, 60, -1)
        self.image = self.images[0]

    def set_initial_position(self):
        self.ptera_height = [height * 0.75, height * 0.65, height * 0.60]
        self.rect.centery = self.ptera_height[random.randrange(0, 2)]
        self.rect.left = self.rect.width + width + random.randrange(0,300)

    def set_velocity(self, speed):
        self.movement = [-1 * speed, 0]

    def draw(self):
        screen.blit(self.image, self.rect)

    def update(self):
        if self.counter % 5 == 0: # Speed of transitions
            self.index = (self.index + 1) % 5 # 5 frame animation
        if self.rect.right <= -1:
            self.kill()
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter +=1
