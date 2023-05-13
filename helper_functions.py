import os
import pygame

scr_size = (width, height) = (854, 480)

def load_image(name, scaled_width=-1, scaled_height=-1, transparent_color=None):
    fullname = os.path.join('sprites', name)
    loaded_image  = pygame.image.load(fullname).convert()

    if scaled_width != -1 or scaled_height != -1:
        loaded_image = pygame.transform.scale(loaded_image, (scaled_width, scaled_height))

    if transparent_color:
        transparent_color = loaded_image.get_at((0, 0)) if transparent_color== -1 else -1
        loaded_image.set_colorkey(transparent_color, pygame.RLEACCEL)


    return (loaded_image , loaded_image.get_rect())

def load_individual_sprites(filenames, scaled_width=-1, scaled_height=-1, transparent_color=None):
    sprites = []
    for filename in filenames:
        fullname = os.path.join('sprites', filename)
        image = pygame.image.load(fullname).convert()

        if transparent_color:
            color_key = image.get_at((0, 0)) if transparent_color == -1 else transparent_color
            image.set_colorkey(color_key, pygame.RLEACCEL)

        if scaled_width != -1 or scaled_height != -1:
            image = pygame.transform.scale(image, (scaled_width, scaled_height))

        sprites.append(image)

    return sprites, sprites[0].get_rect()


def load_sprite_sheet(sheetname, nx, ny, scalex=-1, scaley=-1, transparent_color=None):
    global screen
    fullname = os.path.join('sprites', sheetname)
    sheet = pygame.image.load(fullname)

    if not pygame.display.get_init():
        pygame.display.init()
        screen = pygame.display.set_mode(scr_size)
        sheet = sheet.convert()

    sheet_rect = sheet.get_rect()
    sprites = []

    sprite_width = sheet_rect.width / nx
    sprite_height = sheet_rect.height / ny

    for row in range(ny):
        for col in range(nx):
            sprite_rect = pygame.Rect((col * sprite_width, row * sprite_height, sprite_width, sprite_height))
            sprite_image = pygame.Surface(sprite_rect.size).convert()
            sprite_image.blit(sheet, (0, 0), sprite_rect)

            if transparent_color:
                color_key = sprite_image.get_at((0, 0)) if transparent_color == -1 else transparent_color
                sprite_image.set_colorkey(color_key, pygame.RLEACCEL)

            if scalex != -1 or scaley != -1:
                sprite_image = pygame.transform.scale(sprite_image, (scalex, scaley))

            sprites.append(sprite_image)

    return sprites, sprites[0].get_rect()
