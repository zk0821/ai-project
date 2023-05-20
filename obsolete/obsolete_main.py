# External imports
import sys
import numpy as np
import pygame
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
# Q Learning Bot
from obsolete_rl_bot import Bot, actions
# Deep Q-Learning Bot
from deep_q_learning.agent import Agent
# Game Objects
from game_objects.runner import Runner
from game_objects.cactus import Cactus
from game_objects.bird import Bird
from game_objects.ground import Ground
from game_objects.scoreboard import Scoreboard
from game_objects.death_counter import DeathCounter
from game_objects.warning import TrainingWarning

# Init
sys.setrecursionlimit(100_000)
pygame.init()

scr_size = (width, height) = (854, 480)
FPS = 60

game_quit = False
high_score = 0
death_count = 0
q_learning_training = False
deep_q_learning_training = False
previous_move = 0

screen = pygame.display.set_mode(scr_size)
clock = pygame.time.Clock()
pygame.display.set_caption("DUNE RUNNER 2023")

# Sounds
background_sound = pygame.mixer.Sound('sounds/caravan.ogg.ogg')
background_sound.play(-1)
background_sound.set_volume(0)
jump_sound = pygame.mixer.Sound('sounds/jump_sound_effect.mp3')
jump_sound.set_volume(0)
die_sound = pygame.mixer.Sound('sounds/death_sound_effect.wav')
die_sound.set_volume(0)
slide_sound = pygame.mixer.Sound('sounds/slide_sound_effect.mp3')
slide_sound.set_volume(0)

# Seaborn
sns.set_style("whitegrid")

q_learning_bot = Bot()
deep_q_learning_agent = Agent()

def get_first(cacti, birds):
    next_cacti = get_next_obstacle(cacti)
    next_bird = get_next_obstacle(birds)

    if next_bird and next_cacti:
        return (1, next_bird) if next_bird.rect.x < next_cacti.rect.x else (0, next_cacti)

    return (1, next_bird) if next_bird else (0, next_cacti)


def get_next_obstacle(obstacles):
    next_obstacles = [obstacle for obstacle in obstacles if obstacle.rect.x > 0]
    return next_obstacles[0] if len(next_obstacles) > 0 else None


def get_game_param(player, cacti, birds):
    type_obstacle, next_obstacle = get_first(cacti, birds)
    return ((-player.rect.x + next_obstacle.rect.x), next_obstacle.rect.y, type_obstacle, next_obstacle) if next_obstacle else (0, 0, None, None)


def gameplay():
    global high_score, rl_bot, game_quit, death_count, q_learning_training, deep_q_learning_training, previous_move

    # if q_learning_training:
    #     print("[Q-Learning] Games played: ", q_learning_bot.games_played)
    # elif deep_q_learning_training:
    #     print("[Deep Q-Learning] Games played: ", deep_q_learning_agent.n_game)

    game_speed = 4
    game_over = False
    runner = Runner(50, 50)
    new_ground = Ground(-1 * game_speed)
    scoreboard = Scoreboard()
    death_counter = DeathCounter()
    training_warning = TrainingWarning()
    counter = 0
    level_points = [200, 400, 600, 800, 1000, 1200, 1400]
    level_point_times = {point: None for point in level_points}
    start_time = time.time()


    cacti = pygame.sprite.Group()
    birds = pygame.sprite.Group()
    last_obstacle = pygame.sprite.Group()

    Cactus.containers = cacti
    Bird.containers = birds

    while not game_quit:
        # Until runner dies
        runner.action_cooldown=0
        while not game_over:
            override_event = None
            if pygame.display.get_surface():
                # User Inputs
                user_inputs = filter(lambda current_event: current_event.type == pygame.QUIT or current_event.type == pygame.KEYDOWN, pygame.event.get())
                for event in user_inputs:
                    if event.type == pygame.QUIT:
                        game_quit = True
                        game_over = True

                    elif event.type == pygame.KEYDOWN:
                        # Jump
                        if event.key == pygame.K_UP:
                            override_event = 1
                            # Only if on the ground
                            if runner.rect.bottom == int(0.90 * height):
                                if pygame.mixer.get_init() is not None:
                                    jump_sound.play()
                                runner.movement[1] = -1 * runner.jumpSpeed
                                runner.isJumping = True
                        # Slide
                        elif event.key == pygame.K_DOWN:
                            override_event = 2
                            if not (runner.isJumping and not runner.isDead):
                                runner.isDucking = True
                                slide_sound.play()
                        # Toggle q-learning training
                        elif event.key == pygame.K_RETURN:
                            q_learning_training = not q_learning_training
                            print("Q-Learning training has been turned ON!") if q_learning_training else print("Q-Learning training has been turned OFF!")
                        # Toggle deep q-learning training
                        elif event.key == pygame.K_SPACE:
                            deep_q_learning_training = not deep_q_learning_training
                            print("Deep Q-Learning training has been turned ON!") if deep_q_learning_training else print("Deep Q-Learning training has been turned OFF!")

                # Q-Learning Inputs
                if q_learning_training:
                    # Get bot action
                    bot_event = 0
                    x_diff, obstacle_y, type_obstacle, next_obstacle = get_game_param(runner, cacti, birds)
                    if next_obstacle:
                        bot_event = q_learning_bot.act(x_diff, obstacle_y, game_speed, obstacle=type_obstacle)
                    # Jump
                    if bot_event == actions['UP']:
                        # Only if on the ground
                        if runner.rect.bottom == int(0.90 * height):
                            if pygame.mixer.get_init():
                                jump_sound.play()
                            runner.movement[1] = -1 * runner.jumpSpeed
                            runner.isJumping = True
                            q_learning_bot.action_cooldown = 100
                    # Slide
                    if bot_event == actions['DOWN']:
                        if not (runner.isJumping and not runner.isDead):
                            slide_sound.play()
                            runner.isDucking = True
                            q_learning_bot.action_cooldown = 100

                # Deep Q-Learning Inputs
                if deep_q_learning_training:
                    agent_move = 0
                    # Get old state
                    x_diff, obstacle_y, type_obstacle, next_obstacle = get_game_param(runner, cacti, birds)
                    if type_obstacle is not None:
                        deep_q_learning_agent.current_state = deep_q_learning_agent.get_state([x_diff, obstacle_y, type_obstacle, game_speed])
                        # Decide the action
                        agent_move = deep_q_learning_agent.decide_action()
                    # Jump
                    if agent_move == actions['UP']:
                        # Only if on the ground
                        if runner.rect.bottom == int(0.90 * height):
                            if pygame.mixer.get_init():
                                jump_sound.play()
                            runner.movement[1] = -1 * runner.jumpSpeed
                            runner.isJumping = True
                            deep_q_learning_agent.action_cooldown = 100
                    # Slide
                    if agent_move == actions['DOWN']:
                        if not (runner.isJumping and not runner.isDead):
                            slide_sound.play()
                            runner.isDucking = True
                            deep_q_learning_agent.action_cooldown = 100
            else:
                print("Problems when loading display surface")
                game_quit = True
                game_over = True

            # Check for collision with cacti
            for cactus in cacti:
                cactus.movement[0] = -1 * game_speed
                if pygame.sprite.collide_mask(runner, cactus):
                    if pygame.mixer.get_init() is not None:
                        die_sound.play()
                    runner.isDead = True

            # Check for collision with birds
            for bird in birds:
                bird.movement[0] = -1 * game_speed
                if pygame.sprite.collide_mask(runner, bird):
                    if pygame.mixer.get_init() is not None:
                        die_sound.play()
                    runner.isDead = True

            min_obstacle_distance = 300
            # Cacti Generator
            if len(cacti) < 2:
                # If no cacti -> generate a cacti
                if len(cacti) == 0:
                    if len(last_obstacle) > 0:
                        for l in last_obstacle:
                            distance_to_next_obstacle = width - l.rect.right
                            if distance_to_next_obstacle >= min_obstacle_distance:
                                last_obstacle.empty()
                                last_obstacle.add(Cactus(game_speed))
                    else:
                        last_obstacle.empty()
                        last_obstacle.add(Cactus(game_speed))
                # If a single cacti -> use special method to generate new one
                else:
                    for l in last_obstacle:
                        distance_to_next_obstacle = width - l.rect.right
                        if distance_to_next_obstacle >= min_obstacle_distance and random.randrange(0, 50) == 10:
                            last_obstacle.empty()
                            last_obstacle.add(Cactus(game_speed))

            # Bird generator
            if len(birds) == 0 and random.randrange(0, 1_000_000) < 20 + counter*10 and counter > 10:
                for l in last_obstacle:
                    distance_to_next_obstacle = width - l.rect.right
                    if distance_to_next_obstacle >= min_obstacle_distance:
                        last_obstacle.empty()
                        last_obstacle.add(Bird(game_speed))

            # Make sure that obstacles are more than 100px apart
            # if len(cacti) > 0 and len(birds) > 0 and counter % 2 == 0:
            #     min_distance = 300
            #     last_cactus = cacti.sprites()[-1]
            #     last_bird = birds.sprites()[-1]
            #     if abs(last_cactus.rect.x - last_bird.rect.x) < min_distance:
            #          last_bird.kill()

            # Update the sprites
            runner.update()
            cacti.update()
            birds.update()
            new_ground.update()
            scoreboard.update(runner.score, high_score)
            death_counter.update(death_count)
            training_warning.update(q_learning_training, deep_q_learning_training)

            # Update the bot
            x_diff, y_obstacle, type_obstacle, _ = get_game_param(runner, cacti, birds)
            game_param = x_diff, y_obstacle, game_speed, type_obstacle

            # Re-draw the sprites
            if pygame.display.get_surface() is not None:
                # Needs to be drawn first!
                new_ground.draw()

                runner.draw()
                cacti.draw(screen)
                birds.draw(screen)
                scoreboard.draw()
                death_counter.draw()
                training_warning.draw()

                pygame.display.update()

            # Check if runner died
            if runner.isDead:
                game_over = True
                death_count += 1

                # Update Q-Values and plots only if training mode is enabled
                if q_learning_training:
                    q_learning_bot.update_q_values(game_param, is_dead=runner.isDead, user_event=override_event)
                    q_learning_bot.last_action = 0
                    q_learning_bot.scores_per_game.append(runner.score)

                    # Saving a plot of scores per game
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
                    sns.lineplot(x=range(1, len(q_learning_bot.scores_per_game) + 1), y=q_learning_bot.scores_per_game, label='Scores per game', ax=ax, marker="o")
                    ax.set_xlabel('Number of games', fontsize="14")
                    ax.set_ylabel('Reached score', fontsize="14")
                    ax.set_title('Q-Learning', fontsize="16")

                    window_size = 10
                    if len(q_learning_bot.scores_per_game) >= window_size:
                        # Saving a plot of rolling averages
                        scores_per_game_array = np.array(q_learning_bot.scores_per_game)
                        rolling_average_rewards = np.convolve(scores_per_game_array, np.ones(window_size), 'valid') / window_size
                        sns.lineplot(x=range(window_size, len(q_learning_bot.scores_per_game) + 1), y=rolling_average_rewards, label=f'Rolling average score (window size = {window_size})', ax=ax, marker="o")

                    # for level_point in level_points:
                    #     if runner.score >= level_point and level_point_times[level_point] is None:
                    #         level_point_times[level_point] = time.time() - start_time
                    #         ax.text(len(q_learning_bot.scores_per_game), level_point, f"Level point {level_point} reached at {level_point_times[level_point]:.1f} seconds")

                    ax.legend(loc="upper left", fontsize="12")
                    fig.tight_layout()
                    fig.savefig('scores_and_rolling_average_reward_plot.png')
                    fig.clf()

                # Update Deep Q-Learning Values
                if deep_q_learning_training:
                    # Get the new state
                    new_x_diff, new_obstacle_y, new_type_obstacle, new_next_obstacle = get_game_param(runner, cacti, birds)
                    if new_type_obstacle is not None:
                        deep_q_learning_agent.next_state = deep_q_learning_agent.get_state([new_x_diff, new_obstacle_y, new_type_obstacle, game_speed])
                        reward = -10
                        print(f"PUNISHING ACTION ({deep_q_learning_agent.chosen_action}): {reward}")
                        # Train short memory
                        deep_q_learning_agent.train_short_memory(reward, runner.isDead)
                        # Remember params
                        deep_q_learning_agent.remember(reward, runner.isDead)
                        # Train long memory
                        deep_q_learning_agent.n_game += 1
                        deep_q_learning_agent.train_long_memory()
                        if runner.score > high_score:
                            deep_q_learning_agent.model.save_model()
                    else:
                        print("NEW TYPE OBSTACLE IS NONE!")
                    # Reset model values
                    deep_q_learning_agent.current_state = []
                    deep_q_learning_agent.next_state = []
                    deep_q_learning_agent.chosen_action = 0
                    deep_q_learning_agent.action_cooldown = 0

                if runner.score > high_score:
                    high_score = runner.score

            clock.tick(FPS)
            counter = (counter + 1)

            # We can reset action cooldown if agent jumped or ducked and it already started running again
            if q_learning_training and q_learning_bot.last_action and (q_learning_bot.last_action > 0) and (q_learning_bot.action_cooldown > 0) and (not runner.isJumping) and (not runner.isDucking):
                print(q_learning_bot.action_cooldown)
                q_learning_bot.action_cooldown = 0
                # Reward if not dead and rewarding enabled
                if not runner.isDead and q_learning_bot.rewarding_enabled:
                    q_learning_bot.update_q_values(game_param, is_dead=runner.isDead, user_event=override_event)

            # Update the Deep Q-Learning agent
            if deep_q_learning_training and (deep_q_learning_agent.chosen_action > 0) and (deep_q_learning_agent.action_cooldown > 0) and (not runner.isJumping) and (not runner.isDucking):
                # Reset action cooldown
                deep_q_learning_agent.action_cooldown = 0
                if not runner.isDead:
                    # Get the new state
                    new_x_diff, new_obstacle_y, new_type_obstacle, new_next_obstacle = get_game_param(runner, cacti, birds)
                    if new_type_obstacle is not None:
                        deep_q_learning_agent.next_state = deep_q_learning_agent.get_state([new_x_diff, new_obstacle_y, new_type_obstacle, game_speed])
                        reward = 10
                        print(f"REWARDING ACTION ({deep_q_learning_agent.chosen_action}): {reward}")
                        # Train short memory
                        deep_q_learning_agent.train_short_memory(reward, runner.isDead)
                        # Remember params
                        deep_q_learning_agent.remember(reward, runner.isDead)
                    else:
                        print("NEW TYPE OBSTACLE IS NONE!")

            # Increase game speed
            if counter % 1400 == 0 and game_speed < 11:
                game_speed +=1
                runner.gravity +=0.1
                # Change speed of all relavant game objects
                new_ground.speed = -1 * game_speed
                for cactus in cacti:
                    cactus.set_velocity(game_speed)
                for bird in birds:
                    bird.set_velocity(game_speed)

            if game_over and not game_quit: gameplay()
            if game_quit: break

        game_quit = True
        pygame.quit()
        quit()

def main():
    while not game_quit:
        gameplay()

if __name__ == '__main__':
    main()
