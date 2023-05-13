from itertools import cycle
import pygame
from helper_functions import load_individual_sprites

width, height = (854, 480)
screen = pygame.display.set_mode((width, height))
pygame.mixer.init()
checkpoint_sound = pygame.mixer.Sound('sounds/checkpoint_sound_effect.wav')

class Runner():
    def __init__(self, sizex=-1, sizey=-1):
        self.images, self.rect = load_individual_sprites([f'walk_{i}.png' for i in range(10)], 100, 100, -1)
        self.images1, self.rect1 = load_individual_sprites([f'slide_{i}.png' for i in range(10)], 80, 80, -1)
        self.rect.bottom = int(0.9 * height)
        self.rect.left = width / 50
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.gravity = 0.75
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0, 0]
        self.jumpSpeed = 18

        # for duck action
        self.duck_max = 30
        self.duck_counter = cycle(range(self.duck_max + 1))

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width

    def draw(self):
        screen.blit(self.image, self.rect)

    def checkbounds(self):
        if self.rect.bottom > int(0.90 * height):
            self.rect.bottom = int(0.90 * height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + self.gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1) % 2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1) % 2

        elif self.isDucking:
            if next(self.duck_counter) == self.duck_max:
                self.isDucking = False

            if self.counter % 7 == 0:
                self.index = (self.index + 1) % 9

        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1) % 9 


        if self.isDead:
            self.index = 5

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
            if not self.isJumping:
                self.rect.bottom = int(0.90 * height)

        else:
            self.image = self.images1[self.index % 9]
            self.rect.width = self.duck_pos_width
            self.rect.bottom = int(0.93 * height) # Move player a bit down when ducking

        if not self.isDead and self.counter % 7 == 6 and not self.isBlinking:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0:
                if pygame.mixer.get_init() is not None:
                    checkpoint_sound.play()

        self.counter = (self.counter + 1)