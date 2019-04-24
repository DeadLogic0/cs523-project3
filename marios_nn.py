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

mario_height = 60
mario_width = 42
mario_right_sprites = [pygame.image.load(
            'sprites\\mario\\mario-right\\mario'+str(i+1)+'.png') for i in range(6)]
mario_right_sprites = [pygame.transform.scale(mario_right_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
mario_left_sprites = [pygame.image.load(
            'sprites\\mario\\mario-left\\mario'+str(i+1)+'.png') for i in range(6)]
mario_left_sprites = [pygame.transform.scale(mario_left_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
mario_right_jump = pygame.image.load('sprites\\mario\\mario-right\\mariojump.png')
mario_right_jump = pygame.transform.scale(mario_right_jump,(mario_width,mario_height))
mario_left_jump = pygame.image.load('sprites\\mario\\mario-left\\mariojump.png')
mario_left_jump = pygame.transform.scale(mario_left_jump,(mario_width,mario_height))
luigi_right_sprites = [pygame.image.load(
            'sprites\\luigi\\luigi-right\\luigi'+str(i+1)+'.png') for i in range(6)]
luigi_right_sprites = [pygame.transform.scale(luigi_right_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
luigi_left_sprites = [pygame.image.load(
            'sprites\\luigi\\luigi-left\\luigi'+str(i+1)+'.png') for i in range(6)]
luigi_left_sprites = [pygame.transform.scale(luigi_left_sprites[i],
                        (mario_width,mario_height)) for i in range(6)]
luigi_right_jump = pygame.image.load('sprites\\luigi\\luigi-right\\luigijump.png')
luigi_right_jump = pygame.transform.scale(luigi_right_jump,(mario_width,mario_height))
luigi_left_jump = pygame.image.load('sprites\\luigi\\luigi-left\\luigijump.png')
luigi_left_jump = pygame.transform.scale(luigi_left_jump,(mario_width,mario_height))


mario_max_x_vel = 500#250
mario_y_up_accel = 500
mario_y_down_accel = 50
mario_x_accel = 20
num_of_marios = 200
mario_defeat_bounce = .75


player = False
displayArena = False
displayBest = True
save_gens = True

fps = 60
gravity = 750
x_vel_slow = 1 - 0.8 / fps
x_vel_round_digit = 2

arena_len = 1500
arena_height = 700
arena_floor = 1
arena_leftwall = 40
arena_rightwall = arena_len - mario_width
arena_max_duration = 1500
arena_move_polling_rate = 15
ground_width = 40
random_y_max = mario_height*2
ground_color = (139,69,19)
ground_rect = pygame.Rect(0,
        arena_height,
        arena_len,
        ground_width)

background_color = (0,108,170)

num_of_input_nodes = 6 #12 if inputing velocities
num_of_layer1_nodes = 10 #14 if inputting velocities
num_of_layer2_nodes = 6
mario_nn_layer1 = np.array([[[random.random()*2 - 1 for i in range(num_of_input_nodes)]
                                for i2 in range(num_of_layer1_nodes)]
                                    for i3 in range(num_of_marios)])
mario_nn_layer2 = np.array([[[random.random()*2 - 1 for i in range(num_of_layer1_nodes)]
                                for i2 in range(num_of_layer2_nodes)]
                                    for i3 in range(num_of_marios)])
mutation_prob = 6/(num_of_layer1_nodes + num_of_layer2_nodes)
crossover_probability = 0.4
num_of_tourn_select = 8
num_of_best_to_select = 30
num_of_gen = 100
num_of_matches = 3
jump_threshold = 300
move_threshold = 300

stop = False

file_names = "best"

pygame.init()
clock = pygame.time.Clock()
display = pygame.display.set_mode((arena_len,arena_height+ground_width))
pygame.display.set_caption('marios')


#input is a list of ID's for each mario
def mario_fight(marios):
    global stop
    num_of_marios_fighting = len(marios);
    arena_spacing = arena_rightwall/(num_of_marios_fighting-1)
    arena_spacing -= random.randint(0,math.floor(arena_spacing/2))
    mario_xs = np.array([arena_leftwall+i*arena_spacing for
            i in range(num_of_marios_fighting)], dtype=np.float);
    mario_ys = np.array([arena_floor+random.randint(0,random_y_max) for
            a in range(num_of_marios_fighting)], dtype=np.float)
    mario_x_vel = np.array([0]*num_of_marios_fighting, dtype=np.float)
    mario_y_vel = np.array([0]*num_of_marios_fighting, dtype=np.float)
    mario_status = np.array([True]*num_of_marios_fighting)
    mario_score = np.array([0]*num_of_marios_fighting)
    IDs = [n for n in range(num_of_marios_fighting)]

    playerxmod = 0
    playerymod = 0
    for step in range(arena_max_duration):
        if stop == True: break
        if(displayArena == True):
            display.fill(background_color)
            pygame.draw.rect(display,ground_color,ground_rect,0)
            pygame.draw.rect(display,(40,160,40),pygame.Rect(0,
                    0, arena_leftwall, arena_height),0)
            pygame.draw.rect(display,(40,160,40),pygame.Rect(arena_rightwall+mario_width,
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
        new_mario_ys = np.array([0]*num_of_marios_fighting, dtype=np.float)
        for i in range(num_of_marios_fighting):
            if(mario_status[i] == False): continue
            mario_xs[i] = mario_xs[i] + mario_x_vel[i] / fps
            new_mario_ys[i] = mario_ys[i] + mario_y_vel[i] / fps
            mario_x_vel[i] = round(mario_x_vel[i] * x_vel_slow , x_vel_round_digit)
            if(mario_ys[i] != 0):
                mario_y_vel[i] -= gravity/fps
            if(mario_xs[i] < arena_leftwall):
                mario_xs[i] = arena_leftwall
                mario_x_vel[i] = -1
            elif(mario_xs[i] > arena_rightwall):
                mario_xs[i] = arena_rightwall
                mario_x_vel[i] = 1
            if(new_mario_ys[i] < arena_floor):
                new_mario_ys[i] = arena_floor
                mario_y_vel[i] = 0

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
                        for a in range(len(IDs)):
                            if(IDs[a] == i):
                                IDs = np.delete(IDs,a)
                                break
                        mario_y_vel[i2] = mario_y_up_accel * mario_defeat_bounce
                        mario_score[i2] += 1
                        if(len(IDs) == 1): return mario_score
                        break

        mario_ys = new_mario_ys
        _ , ID = (list(l) for l in zip(*sorted(zip(mario_xs[IDs],IDs))))
        for i in range(len(ID)):
            id = ID[i]
            if(id == 0 and player == True): continue
            if(step % arena_move_polling_rate != 0):
                0
            elif(i == 0):
                xmod,ymod = get_move(marios[id] , np.array([mario_xs[ id ]   , mario_ys[ id ],
                                        mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ],
                                        0                   ,     0]))
            elif(i == len(ID) - 1):
                xmod,ymod = get_move(marios[id] , np.array([mario_xs[ id ]   , mario_ys[ id ],
                                         0                   ,     0 ,
                                        mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ]]))
            else:
                xmod,ymod = get_move(marios[id] , np.array([mario_xs[ id ]   , mario_ys[ id ],
                                                  mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ],
                                                  mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ]]))
            # elif(i == 0):#used for inputting velocities but ai wouldnt train fast enough
            #     xmod,ymod = get_move(marios[id] , np.array([mario_xs[ id ]   , mario_ys[ id ], mario_x_vel[id], mario_y_vel[id],
            #                             mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ], mario_x_vel[ ID[i+1] ], mario_y_vel[ ID[i+1] ],
            #                             0                   ,     0              ,        0              ,            0]))
            # elif(i == len(ID) - 1):
            #     xmod,ymod = get_move(marios[id] , np.array([mario_xs[ id ]   , mario_ys[ id ], mario_x_vel[id], mario_y_vel[id],
            #                              0                   ,     0            ,    0     ,         0,
            #                             mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ], mario_x_vel[ ID[i-1] ], mario_y_vel[ ID[i-1] ]]))
            # else:
            #     xmod,ymod = get_move(marios[id] , np.array([mario_xs[ id ]   , mario_ys[ id ], mario_x_vel[id], mario_y_vel[id],
            #                                       mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ], mario_x_vel[ ID[i+1] ], mario_y_vel[ ID[i+1] ],
            #                                       mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ], mario_x_vel[ ID[i-1] ], mario_y_vel[ ID[i-1] ]]))
            mario_x_vel[id] += xmod*mario_x_accel
            if(ymod == 1 and mario_ys[id] == arena_floor):
                mario_y_vel[id] = mario_y_up_accel
            elif(playerymod == -1 and mario_ys[id] != arena_floor):
                mario_y_vel[id] -= mario_y_down_accel
            if(abs(mario_x_vel[id]) >= mario_max_x_vel):
                mario_x_vel[id] = xmod*mario_max_x_vel
            if(displayArena == True and player == False):
                display_mario(step, marios[id], mario_xs[id], mario_ys[id], mario_x_vel[id], mario_y_vel[id])
            if(displayArena == True and player == True):
                display_luigi(step, marios[id], mario_xs[id], mario_ys[id], mario_x_vel[id], mario_y_vel[id])

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

        if(displayArena == True):
            pygame.display.update()
            clock.tick(fps)
    return mario_score

def get_move(marioID, marios):
    move = np.array([a if a >= 0 else 0 for a in  mario_nn_layer1[marioID].dot(marios)])
    move = np.array([a if a >= 0 else 0 for a in  mario_nn_layer2[marioID].dot(move)])
    # move = mario_nn_layer1[marioID].dot(marios)
    # move = mario_nn_layer2[marioID].dot(move)
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

def display_mario(step, marioID, x, y, xvel, yvel):
    global display
    if(y != arena_floor):
        if(xvel >= 0):
            display.blit(mario_right_jump,(x,arena_height - y - mario_height))
        else:
            display.blit(mario_left_jump,(x,arena_height - y - mario_height))
    else:
        if(round(xvel) == 0):
            display.blit(mario_right_sprites[0],(x,arena_height - y - mario_height))
        elif(xvel > 0):
            display.blit(mario_right_sprites[step%6],(x,arena_height - y - mario_height))
        else:
            display.blit(mario_left_sprites[step%6],(x,arena_height - y - mario_height))

def display_luigi(step, marioID, x, y, xvel, yvel):
    global display
    if(y != arena_floor):
        if(xvel >= 0):
            display.blit(luigi_right_jump,(x,arena_height - y - mario_height))
        else:
            display.blit(luigi_left_jump,(x,arena_height - y - mario_height))
    else:
        if(round(xvel) == 0):
            display.blit(mario_right_sprites[0],(x,arena_height - y - mario_height))
        elif(xvel > 0):
            display.blit(mario_right_sprites[step%6],(x,arena_height - y - mario_height))
        else:
            display.blit(luigi_left_sprites[step%6],(x,arena_height - y - mario_height))

def mutation(best):
    global mario_nn_layer1
    global mario_nn_layer2
    for i in range(num_of_marios):
        if(i in best): continue
        for i2 in range(num_of_layer1_nodes):
            if(random.random() > mutation_prob):
                mario_nn_layer1[i][i2] = [random.random()*2 - 1 for a in range(num_of_input_nodes)]
        for i2 in range(num_of_layer2_nodes):
            if(random.random() > mutation_prob):
                mario_nn_layer2[i][i2] = [random.random()*2 - 1 for a in range(num_of_layer1_nodes)]

def crossover(best):
    global mario_nn_layer1
    global mario_nn_layer2
    for i in range(num_of_marios):
        if(i in best): continue
        if(random.random() < crossover_probability):
            r = random.sample(best,1)[0]
            mario_nn_layer1[i] = np.copy(mario_nn_layer1[r])
            mario_nn_layer2[i] = np.copy(mario_nn_layer2[r])
            continue
        best_2 = random.sample(best,2)
        cx_point = random.randint(1,num_of_layer1_nodes + num_of_layer2_nodes-1)
        if(cx_point < num_of_layer1_nodes):
            mario_nn_layer1[i] = np.array([mario_nn_layer1[best_2[0]][c] if c < cx_point
                                    else mario_nn_layer1[best_2[1]][c] for c in range(num_of_layer1_nodes)])
            mario_nn_layer2[i] = np.copy(mario_nn_layer2[best_2[1]])
        else:
            mario_nn_layer1[i] = np.copy(mario_nn_layer1[best_2[1]])
            mario_nn_layer2[i] = np.array([mario_nn_layer2[best_2[0]][c] if c < cx_point
                                    else mario_nn_layer2[best_2[1]][c] for c in range(num_of_layer2_nodes)])


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
    path = "best_nn"
    try:
        os.mkdir(path)
    except OSError:
        0
    else:
        0
    for gen in range(num_of_gen):
        fps = 20
        arena_max_duration = 100*fps
        arena_move_polling_rate = 3
        rand = (arena_len - random.randint(700,1480))/2
        arena_rightwall = arena_len - rand - mario_width
        arena_leftwall = rand
        print("gen: "+str(gen))
        if(stop == True): break
        best = [0]*num_of_best_to_select
        scores = [0]*num_of_best_to_select
        for i in range(num_of_best_to_select):
            ids = [a for a in range(num_of_tourn_select)]
            results = np.array([0]*num_of_tourn_select)
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
        best = numpy.unique(best).tolist()
        scores , best = (list(l) for l in zip(*sorted(zip(scores , best))))
        best = [best[i] for i in range(len(best)) if scores[i] != 0 ]
        scores = [scores[i] for i in range(len(scores)) if scores[i] != 0 ]
        if(len(best) < 2): best = random.sample(range(0,num_of_marios),num_of_tourn_select)
        print("Num of Unique Best:  "+ str(len(best)))
        print("Average Defeated Marios:  "+ str(np.mean(scores)/num_of_matches)+"\n")
        if(displayBest == True):
            displayArena = True
            fps = 30
            arena_max_duration = 750
            arena_move_polling_rate = 3
            #player = True
            if(len(best) < num_of_tourn_select):
                mario_fight(best)
            else:
                mario_fight(best[len(best)-num_of_tourn_select:len(best)])
            display.fill(background_color)
            pygame.draw.rect(display,ground_color,ground_rect,0)
            #player = False
            displayArena = False
        if(gen < num_of_gen):
            crossover(best)
            mutation(best)
        path = "best_nn\\gen"+str(gen)+"\\"
        try:
            os.mkdir(path)
        except OSError:
            0
        else:
            0
        if(save_gens == True):
            for i in range(len(best)):
                np.savetxt(path+file_names+str(i)+'_1.dat',mario_nn_layer1[best[0]])
                np.savetxt(path+file_names+str(i)+'_2.dat',mario_nn_layer2[best[0]])

def main():
    genetic_algorithm()
    pygame.quit()

if __name__ == "__main__":
    main()
