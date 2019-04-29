"""
UNM CS523 Spring 2019
Project 3
Author: Nicholas Barrett
"""

from __future__ import division
import numpy as np
import random
import numpy
import time
import pygame
import os
import math
from multiprocessing import Pool
from multiprocessing.dummy import Pool as Pool2

"""
Setting size of mario_width and mario_height needed for collision detection and sprite resizing
"""
mario_height = 60
mario_width = 42

"""
Mario and Luigi sprite loading and resizing to mario_heigth and mario_width
"""
mario_right_sprites = [pygame.image.load(
            'sprites/mario/mario-right/mario'+str(i+1)+'.png') for i in range(6)]
mario_right_sprites = [pygame.transform.scale(mario_right_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
mario_left_sprites = [pygame.image.load(
            'sprites/mario/mario-left/mario'+str(i+1)+'.png') for i in range(6)]
mario_left_sprites = [pygame.transform.scale(mario_left_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
mario_right_jump = pygame.image.load('sprites/mario/mario-right/mariojump.png')
mario_right_jump = pygame.transform.scale(mario_right_jump,(mario_width,mario_height))
mario_left_jump = pygame.image.load('sprites/mario/mario-left/mariojump.png')
mario_left_jump = pygame.transform.scale(mario_left_jump,(mario_width,mario_height))
luigi_right_sprites = [pygame.image.load(
            'sprites/luigi/luigi-right/luigi'+str(i+1)+'.png') for i in range(6)]
luigi_right_sprites = [pygame.transform.scale(luigi_right_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
luigi_left_sprites = [pygame.image.load(
            'sprites/luigi/luigi-left/luigi'+str(i+1)+'.png') for i in range(6)]
luigi_left_sprites = [pygame.transform.scale(luigi_left_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
luigi_right_jump = pygame.image.load('sprites/luigi/luigi-right/luigijump.png')
luigi_right_jump = pygame.transform.scale(luigi_right_jump,(mario_width,mario_height))
luigi_left_jump = pygame.image.load('sprites/luigi/luigi-left/luigijump.png')
luigi_left_jump = pygame.transform.scale(luigi_left_jump,(mario_width,mario_height))

"""
Mario movement characteristic variables
"""
mario_max_x_vel = 500 #max x velocity
mario_y_up_accel = 500 #jump veloctiy
mario_y_down_accel = 10 #down acceleration
mario_x_accel = 20 #x acceleration
mario_defeat_bounce = .75 #bounce velecity = mario_defeat_bounce * mario_y_up_accel

"""
Genetic Algorithm variables
"""
num_of_marios = 150 #number of marios each gen
num_of_mutations = 2 #number of mutations applied to each new neural network
num_of_mutations_per_node = 3 #total mutations - num_of_mutations*num_of_mutations_per_node
crossover_probability = 0.4 #probability of crossover
num_of_tourn_select = 8 #number of marios randomly selected to participate in each tournament
num_of_matches = 3 #number of matches in a tournament
#number of best marios selected each generation, duplicates and 0 fitness removed after selection
num_of_best_to_select = 30 #^^^
num_of_gen = 100 #number of generations


"""
Mario neural network variables
"""

num_of_input_nodes = 14
num_of_layer1_nodes = 6
mario_nn_layer1 = np.array([[[random.random()*2 - 1 for i in range(num_of_input_nodes)]
                                for i2 in range(num_of_layer1_nodes)]
                                    for i3 in range(num_of_marios)])


"""
Data Output Variables and PVE variable
"""
#if player = True the current mario_fight will allow a player to control
#the lone mario on the feild using the up,down,left,right controls
player = False
displayArena = False #if displayArena = True the current mario_fight will be displayed
displayBest = True #if displayBest = True the best of the generation will be displayed fighting
#if save_gens = True the neural network weights will be outputted to a .dat file
#using the file_names variable from below
save_gens = False
file_names = "best" #"file_names"+"mario Index"+"neural network layer"+".dat"


"""
mario_fight variables
"""
fps = 30 #fps of the mario fight, if under ~15 the collision detection wont work properly
gravity = 750 #gravity value
x_vel_slow = 1 - 0.8 / fps #slow applied to mario x velocity each frame
x_vel_round_digit = 0 #x velocity rounded to x_vel_round_digit's
arena_len = 1500 #length of screen and max possible arena length
arena_height = 700 #screen height - ground_height
arena_floor = 1 #position of arena floor, 0 is reserved for instances where there are only two marios
arena_leftwall = 40 #x coordinate of left wall
arena_rightwall = arena_len - mario_width - arena_leftwall #x coordinate of right wall
arena_max_duration = 1500 #max mario_fight duration
arena_move_polling_rate = 5 #marios choose a new move every arena_move_polling_rate frames
ground_height = 40 #height of rect representing the ground
random_y_max = mario_height-1 #random y max position
ground_color = (139,69,19) #color of the ground
ground_rect = pygame.Rect(0, #ground rectangle
        arena_height,
        arena_len,
        ground_height)
background_color = (0,108,170) #background color
wall_color = (40,160,40) #background color
random_wrap = False
wall_death_weight = -2
wall_collision_weight = -.1
wrap = False
wall_deadly = False
random_arena_size = True

"""
pygame gui variables
"""
stop = False
pygame.init()
clock = pygame.time.Clock()
display = pygame.display.set_mode((arena_len,arena_height+ground_height))
pygame.display.set_caption('marios')

"""
load previous generation
"""
def load_gen(gen, num_of_marios):
    global mario_nn_layer1
    path = "best_nn/gen"+str(gen)+"/"
    for i in range(num_of_marios):
        mario_nn_layer1[i] = np.loadtxt(path+file_names+str(i)+"_1.dat")

"""
mario fight match
inputs is a list of mario indexs
marios are positions left to right in the same order as the inputs
"""
def mario_fight(marios):
    global stop
    """
    game variables
    """
    num_of_marios_fighting = len(marios);
    arena_spacing = (arena_rightwall - arena_leftwall)/(num_of_marios_fighting-1)
    arena_spacing -= random.randint(0,math.floor(arena_spacing/2))
    mario_xs = np.array([arena_leftwall+i*arena_spacing for
            i in range(num_of_marios_fighting)], dtype=np.float);
    mario_ys = np.array([arena_floor+random.randint(0,random_y_max) for
            a in range(num_of_marios_fighting)], dtype=np.float)
    mario_x_vel = np.array([0]*num_of_marios_fighting, dtype=np.float)
    mario_y_vel = np.array([0]*num_of_marios_fighting, dtype=np.float)
    mario_status = np.array([True]*num_of_marios_fighting)
    mario_score = np.array([0]*num_of_marios_fighting,dtype=np.float)
    mario_survival = np.array([0]*num_of_marios_fighting, dtype=np.float)
    IDs = [n for n in range(num_of_marios_fighting)]

    playerxmod = 0 #player controls variable
    playerymod = 0 #player controls variable
    for step in range(arena_max_duration):
        """
        get pygame events and update variables
        """
        if(displayArena == True):
            display.fill(background_color)
            pygame.draw.rect(display,ground_color,ground_rect,0)
            if(wrap == False):
                pygame.draw.rect(display,wall_color,pygame.Rect(0,
                        0, arena_leftwall, arena_height),0)
                pygame.draw.rect(display,wall_color,pygame.Rect(arena_rightwall+mario_width,
                    0, 1000, arena_height),0)
            elif(wrap == True):
                pygame.draw.rect(display,(0,0,0),pygame.Rect(0,
                        0, arena_leftwall+mario_width, arena_height),0)
                pygame.draw.rect(display,(0,0,0),pygame.Rect(arena_rightwall,
                    0, 1000, arena_height),0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    playerymod = 1
                if event.key == pygame.K_DOWN:
                    playerymod = -1
                if event.key == pygame.K_RIGHT:
                    playerxmod = 1
                if event.key == pygame.K_LEFT:
                    playerxmod = -1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT:
                    playerxmod = 0
                if event.key == pygame.K_DOWN:
                    playerymod = 0
        if stop == True: break

        """
        update the mario x, y, x velocity, and y velocity
        new y positions are saved to a new array for collision detection purposes
        """
        new_mario_ys = np.array([0]*num_of_marios_fighting, dtype=np.float)
        for i in range(num_of_marios_fighting):
            if(mario_status[i] == False): continue
            mario_xs[i] = mario_xs[i] + mario_x_vel[i] / fps
            new_mario_ys[i] = mario_ys[i] + mario_y_vel[i] / fps
            mario_x_vel[i] = round(mario_x_vel[i] * x_vel_slow , x_vel_round_digit)
            if(mario_ys[i] != 0):
                mario_y_vel[i] -= gravity/fps
            if(wrap):
                if(mario_xs[i] - mario_width < arena_leftwall):
                    mario_xs[i] = arena_rightwall - 1
                elif(mario_xs[i] > arena_rightwall):
                    mario_xs[i] = arena_leftwall - mario_width + 1
            else:
                if(wall_deadly == False):
                    if(mario_xs[i] < arena_leftwall):
                        mario_xs[i] = arena_leftwall
                        mario_score[i] += wall_collision_weight
                        mario_x_vel[i] = -1
                    elif(mario_xs[i] > arena_rightwall):
                        mario_xs[i] = arena_rightwall
                        mario_score[i] += wall_collision_weight
                        mario_x_vel[i] = 1
                else:
                    if(mario_xs[i] < arena_leftwall):
                        mario_survival[i] = step/arena_max_duration
                        mario_score[i] += wall_death_weight
                        for a in range(len(IDs)):
                            if(IDs[a] == i):
                                IDs = np.delete(IDs,a)
                                break
                    elif(mario_xs[i] > arena_rightwall):
                        mario_survival[i] = step/arena_max_duration
                        mario_score[i] += wall_death_weight
                        for a in range(len(IDs)):
                            if(IDs[a] == i):
                                IDs = np.delete(IDs,a)
                                break
            if(new_mario_ys[i] < arena_floor):
                new_mario_ys[i] = arena_floor
                mario_y_vel[i] = 0

        """
        collision detection
        """
        for i in range(num_of_marios_fighting):
            if(mario_status[i] == False): continue
            mario_x = mario_xs[i]
            mario_y = new_mario_ys[i]
            for i2 in range(num_of_marios_fighting):
                if(i == i2 or mario_status[i2] == False): continue
                if((mario_xs[i2] >= mario_x and
                        mario_xs[i2] <= mario_x + mario_width) or
                        (mario_xs[i2] + mario_width>= mario_x and
                                mario_xs[i2]  <= mario_x)):
                    if(new_mario_ys[i2] <= mario_y + mario_height and
                            new_mario_ys[i2] > mario_y and
                            mario_ys[i2] > mario_ys[i] + mario_height ):
                        mario_status[i] = False
                        mario_survival[i] = step/arena_max_duration
                        for a in range(len(IDs)):
                            if(IDs[a] == i):
                                IDs = np.delete(IDs,a)
                                break
                        mario_y_vel[i2] = mario_y_up_accel * mario_defeat_bounce
                        mario_score[i2] += 1
                        break
        mario_ys = new_mario_ys
        if(len(IDs) == 0):
            if(np.sum(mario_score) == 0):
                # mario_score = np.array([mario_score[i]*mario_score[i]*mario_survival[i] for i in
                #         range(num_of_marios_fighting) ], dtype=np.float)
                return mario_score
            return mario_score
        if(len(IDs) == 1):
            id = IDs[0]
            if(displayArena == True and player == False):
                display_mario(step, marios[id], mario_xs[id], mario_ys[id], mario_x_vel[id], mario_y_vel[id])
            elif(displayArena == True and player == True and IDs[0] == 0):
                display_mario(step, marios[id], mario_xs[id], mario_ys[id], mario_x_vel[id], mario_y_vel[id])
            elif(displayArena == True and player == True):
                display_luigi(step, marios[id], mario_xs[id], mario_ys[id], mario_x_vel[id], mario_y_vel[id])
            if(displayArena == True):
                pygame.display.update()
                clock.tick(fps)
            return mario_score
        """
        get new moves and update display
        """
        _ , ID = (list(l) for l in zip(*sorted(zip(mario_xs[IDs],IDs))))
        for i in range(len(ID)):
            id = ID[i]
            if(id == 0 and player == True): continue
            if(step % arena_move_polling_rate != 0):
                0
            elif(i == 0):#used for inputting velocities but ai wouldnt train fast enough
                xmod,ymod = get_move(marios[id] , np.array([
                                        arena_leftwall, arena_rightwall,
                                        mario_xs[ id ]   , mario_ys[ id ],
                                        mario_x_vel[id], mario_y_vel[id],
                                        mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ],
                                        mario_x_vel[ ID[i+1] ], mario_y_vel[ ID[i+1] ],
                                        0        ,     0              ,
                                        0        ,     0
                                                ]))
            elif(i == len(ID) - 1):
                xmod,ymod = get_move(marios[id] , np.array([
                                        arena_leftwall, arena_rightwall,
                                        mario_xs[ id ]   , mario_ys[ id ],
                                        mario_x_vel[id], mario_y_vel[id],
                                         0         ,     0            ,
                                         0         ,     0            ,
                                        mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ],
                                        mario_x_vel[ ID[i-1] ], mario_y_vel[ ID[i-1] ],
                                        ]))
            else:
                xmod,ymod = get_move(marios[id] , np.array([
                                        arena_leftwall, arena_rightwall,
                                        mario_xs[ id ]   , mario_ys[ id ],
                                        mario_x_vel[id], mario_y_vel[id],
                                        mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ],
                                        mario_x_vel[ ID[i+1] ], mario_y_vel[ ID[i+1] ],
                                        mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ],
                                        mario_x_vel[ ID[i-1] ], mario_y_vel[ ID[i-1] ]
                                                  ]))
            mario_x_vel[id] += xmod*mario_x_accel
            if(ymod == 1 and mario_ys[id] == arena_floor):
                mario_y_vel[id] = mario_y_up_accel
            elif(ymod == -1 and mario_ys[id] != arena_floor):
                mario_y_vel[id] -= mario_y_down_accel
            if(abs(mario_x_vel[id]) >= mario_max_x_vel):
                mario_x_vel[id] = xmod*mario_max_x_vel
            if(displayArena == True and player == False):
                display_mario(step, marios[id], mario_xs[id], mario_ys[id], mario_x_vel[id], mario_y_vel[id])
            if(displayArena == True and player == True):
                display_luigi(step, marios[id], mario_xs[id], mario_ys[id], mario_x_vel[id], mario_y_vel[id])

        """
        Get player moves if PVE
        """
        if(player == True and IDs[0] == 0):
            mario_x_vel[0] += playerxmod * mario_x_accel
            if(playerymod == 1 and mario_ys[0] == arena_floor):
                mario_y_vel[0] = mario_y_up_accel
                playerymod = 0
            elif(playerymod == -1 and mario_ys[0] != arena_floor):
                mario_y_vel[0] -= mario_y_down_accel
            if(abs(mario_x_vel[0]) >= mario_max_x_vel):
                mario_x_vel[0] = playerxmod * mario_max_x_vel
            display_mario(step, marios[0], mario_xs[0], mario_ys[0], mario_x_vel[0], mario_y_vel[0])

        """
        update display
        """
        if(displayArena == True):
            pygame.display.update()
            clock.tick(fps)
    return mario_score

"""
get move for the mario index using the neural network for that index
"""
def get_move(marioID, marios):
    move = np.array(mario_nn_layer1[marioID].dot(marios))
    returnmove = [0, 0] #default not moving
    #[left,stop,right,down,no jump,jump]
    if(move[1] > move[0] and move[1] > move[2]):
        returnmove[0] = 0 #stop
    elif(move[0] > move[1] and move[0] > move[2]):
        returnmove[0] = -1 #left
    elif(move[2] > move[1] and move[2] > move[0]):
        returnmove[0] = 1 #right
    if(move[4] > move[3] and move[4] > move[5]):
        returnmove[1] = 0 #no jump
    elif(move[3] > move[4] and move[3] > move[5]):
        returnmove[1] = -1 #down
    elif(move[5] > move[3] and move[5] > move[4]):
        returnmove[1] = 1 #jump
    return returnmove

"""
image tinting
"""
def tint(image, marioID):
    image = image.copy()
    mod = marioID/num_of_marios*100
    image.fill((mod*(1 if marioID%2 else 0),
            mod*(1 if marioID%3 else 0),
            mod*(1 if marioID%5 else 0), 100), special_flags=pygame.BLEND_SUB)
    return image

"""
display the mario sprite
"""
def display_mario(step, marioID, x, y, xvel, yvel):
    global display
    if(y != arena_floor):
        if(xvel >= 0):
            display.blit(tint(mario_right_jump,marioID),(x,arena_height - y - mario_height))
        else:
            display.blit(tint(mario_left_jump,marioID),(x,arena_height - y - mario_height))
    else:
        if(round(xvel) == 0):
            display.blit(tint(mario_right_sprites[0],marioID),(x,arena_height - y - mario_height))
        elif(xvel > 0):
            display.blit(tint(mario_right_sprites[step%6],marioID),(x,arena_height - y - mario_height))
        else:
            display.blit(tint(mario_left_sprites[step%6],marioID),(x,arena_height - y - mario_height))

"""
display the luigi sprite
"""
def display_luigi(step, marioID, x, y, xvel, yvel):
    global display
    if(y != arena_floor):
        if(xvel >= 0):
            display.blit(luigi_right_jump,(x,arena_height - y - mario_height))
        else:
            display.blit(luigi_left_jump,(x,arena_height - y - mario_height))
    else:
        if(round(xvel) == 0):
            display.blit(luigi_right_sprites[0],(x,arena_height - y - mario_height))
        elif(xvel > 0):
            display.blit(luigi_right_sprites[step%6],(x,arena_height - y - mario_height))
        else:
            display.blit(luigi_left_sprites[step%6],(x,arena_height - y - mario_height))

"""
perform mutation on any mario neural network not in best
"""
def mutation(best):
    global mario_nn_layer1
    global mario_nn_layer2
    for i in range(num_of_marios):
        if(i in best): continue
        for i2 in range(num_of_mutations):
            ind = random.randint(0,num_of_layer1_nodes-1)
            for i3 in range(num_of_mutations_per_node):
                mario_nn_layer1[i][ind][random.randint(0,
                    num_of_input_nodes-1)] = random.random()*2 - 1

"""
perform crossover on any mario neural network not in best
"""
def crossover(best):
    global mario_nn_layer1
    global mario_nn_layer2
    for i in range(num_of_marios):
        if(i in best): continue
        if(random.random() < crossover_probability):
            r = random.sample(best,1)[0]
            mario_nn_layer1[i] = np.copy(mario_nn_layer1[r])
            continue
        best_2 = random.sample(best,2)
        cx_point = random.randint(1,num_of_layer1_nodes-2)
        mario_nn_layer1[i] = np.array([mario_nn_layer1[best_2[0]][c] if c < cx_point
                                else mario_nn_layer1[best_2[1]][c] for c in range(num_of_layer1_nodes)])

"""
genetic algorithm
"""
def genetic_algorithm():
    global stop
    global displayArena
    global player
    global fps
    global arena_max_duration
    global arena_move_polling_rate
    global arena_len
    global arena_rightwall
    global arena_leftwall
    global wrap
    path = "best_nn"
    """
    make output data directory
    """
    if(save_gens):
        try:
            os.mkdir(path)
        except OSError:
            0
        else:
            0
    print("Gen         Num_Of_Unique_Marios")
    for gen in range(num_of_gen+1):
        """
        randomize arena parameters for generation
        """
        if(random_wrap):
            wrap = random.randint(0,1) == 1
        if(random_arena_size):
            rand = (arena_len - random.randint(700,1400))/2
            arena_rightwall = arena_len - rand - mario_width
            arena_leftwall = rand
        """
        check if pygame gui closed
        """
        if(stop == True): break

        """
        tournament selection
        """
        best = [0]*num_of_best_to_select
        scores = [0]*num_of_best_to_select
        for i in range(num_of_best_to_select):
            ids = [a for a in range(num_of_tourn_select)]
            results = np.array([0]*num_of_tourn_select, dtype=np.float)
            random_marios = random.sample(range(0,num_of_marios),num_of_tourn_select)
            for a in range(num_of_matches):
                if(stop == True): break
                result = mario_fight([random_marios[id] for id in ids])
                ids , result = (list(l) for l in zip(*sorted(zip(ids , result))))
                results += result
                random.shuffle(ids)
            results , IDs = (list(l) for l in zip(*sorted(zip(results , random_marios))))
            best[i] = IDs[num_of_tourn_select-1]
            scores[i] = results[num_of_tourn_select-1]

        """
        remove duplicates and 0 fitness
        """
        best = numpy.unique(best).tolist()
        scores , best = (list(l) for l in zip(*sorted(zip(scores , best))))
        best = [best[i] for i in range(len(best)) if scores[i] != 0 ]
        scores = [scores[i] for i in range(len(scores)) if scores[i] != 0 ]
        if(len(best) < 2): best = random.sample(range(0,num_of_marios),num_of_tourn_select)

        """
        display outputs
        """
        print(str(gen)+"              "+str(len(best)))
        if(displayBest == True):
            displayArena = True
            arena_max_duration = 1500
            # player = True
            if(player != True):
                if(len(best) < num_of_tourn_select):
                    mario_fight(best)
                else:
                    mario_fight(best[len(best)-num_of_tourn_select:len(best)])
            else:
                if(len(best) < num_of_tourn_select):
                    mario_fight([-1]+best)
                else:
                    mario_fight([-1]+best[len(best)-num_of_tourn_select:len(best)])
            arena_max_duration = 750
            # player = False
            displayArena = False
        if(gen < num_of_gen):
            crossover(best)
            mutation(best)

        """
        save the neural networks to output directory
        """
        path = "best_nn/gen"+str(gen)+"/"
        if(save_gens == True):
            try:
                os.mkdir(path)
            except OSError:
                0
            else:
                0
            ind = [a for a in range(len(best)-1,len(best)-num_of_tourn_select-1,-1)]
            for i in range(0,num_of_tourn_select):
                np.savetxt(path+file_names+str(i)+'_1.dat',mario_nn_layer1[best[ind[i]]])

def replay_last_GA():
    global displayArena
    global arena_max_duration
    displayArena = True
    arena_max_duration = 450
    for i in range(0,101):
        load_gen(i,num_of_tourn_select)
        mario_fight([a for a in range(num_of_tourn_select)])

def main():
    replay_last_GA()
    # global displayArena
    # load_gen(5,num_of_tourn_select)
    # displayArena = True
    # mario_fight([a for a in range(num_of_tourn_select)])
    # genetic_algorithm()
    pygame.quit()

if __name__ == "__main__":
    main()
