from __future__ import division
import numpy as np
import random
import numpy
import time
import pygame
from multiprocessing import Pool
from multiprocessing.dummy import Pool as Pool2

mario_height = 70
mario_width = 50
mario_max_x_vel = 250
mario_y_accel = 500
mario_x_accel = 20
num_of_marios = 3
mario_defeat_bounce = .75

fps = 60
gravity = 750
x_vel_slow = 1 - 0.8 / fps
x_vel_round_digit = 10

arena_len = 1000
arena_height = 500
arena_floor = 1
arena_leftwall = 1
arena_rightwall = arena_leftwall + arena_len - mario_width
arena_max_duration = 3000


pygame.init()
clock = pygame.time.Clock()
display = pygame.display.set_mode((arena_len,arena_height))
pygame.display.set_caption('marios')


#input is a list of ID's for each mario
def mario_fight(marios):
    stop = False
    num_of_marios_fighting = len(marios);
    arena_spacing = arena_rightwall/(num_of_marios_fighting-1)
    mario_xs = np.array([arena_leftwall+i*arena_spacing for
            i in range(num_of_marios_fighting)], dtype=np.float);
    mario_ys = np.array([arena_floor]*num_of_marios_fighting, dtype=np.float)
    mario_x_vel = np.array([0]*num_of_marios_fighting, dtype=np.float)
    mario_y_vel = np.array([0]*num_of_marios_fighting, dtype=np.float)
    mario_status = np.array([True]*num_of_marios_fighting)
    mario_score = np.array([0]*num_of_marios_fighting)
    IDs = [n for n in range(num_of_marios_fighting)]

    xmod = 0

    for step in range(arena_max_duration):
        if stop == True: break
        new_mario_ys = np.array([0]*num_of_marios_fighting, dtype=np.float)
        display.fill((0,0,0))
        for i in range(num_of_marios_fighting):
            if(mario_status[i] == False): continue
            mario_xs[i] = mario_xs[i] + mario_x_vel[i] / fps
            new_mario_ys[i] = mario_ys[i] + mario_y_vel[i] / fps
            mario_x_vel[i] = round(mario_x_vel[i] * x_vel_slow , x_vel_round_digit)
            if(mario_ys[i] != 0):
                mario_y_vel[i] -= gravity/fps
            if(mario_xs[i] < arena_leftwall):
                mario_xs[i] = arena_leftwall
                mario_x_vel[i] = 0
            elif(mario_xs[i] > arena_rightwall):
                mario_xs[i] = arena_rightwall
                mario_x_vel[i] = 0
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
                        mario_y_vel[i2] = mario_y_accel*mario_defeat_bounce
                        mario_score[i2] += 1
                        if(len(IDs) == 1): return mario_score
                        break

        mario_ys = new_mario_ys
        _ , ID = (list(l) for l in zip(*sorted(zip(mario_xs[IDs],IDs))))
        for i in range(len(ID)):
            id = ID[i]
            if(i == 0): continue
            # if(i == 0):
            #     xmod,ymod = get_move(np.array([marios[id],
            #                                    mario_xs[ id ]   , mario_ys[ id ],
            #                                    mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ],
            #                                    0                   ,     0]))
            # elif(i == num_of_marios_fighting - 1):
            #     xmod,ymod = get_move(np.array([marios[id],
            #                                    mario_xs[ id ]   , mario_ys[ id ],
            #                                    0                   ,     0            ,
            #                                    mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ]]))
            # else:
            #     xmod,ymod = get_move(np.array([marios[id],
            #                                    mario_xs[ id ]   , mario_ys[ id ],
            #                                    mario_xs[ ID[i+1] ] , mario_ys[ ID[i+1] ],
            #                                    mario_xs[ ID[i-1] ] , mario_ys[ ID[i-1] ]]))
            # mario_x_vel[i] += xmod*mario_x_accel
            # if(ymod == 1 and mario_ys[i] == arena_floor):
            #     mario_y_vel[i] = mario_y_accel
            # if(abs(mario_x_vel[i]) >= mario_max_x_vel):
            #     mario_x_vel[i] = xmod*mario_max_x_vel
            display_mario(mario_xs[id],mario_ys[id])
        i = 0
        id = ID[0]
        ymod = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    ymod = 1
                if event.key == pygame.K_RIGHT:
                    xmod = 1
                if event.key == pygame.K_LEFT:
                    xmod = -1
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT:
                    xmod = 0
        mario_x_vel[i] += xmod*mario_x_accel
        if(ymod == 1 and mario_ys[i] == arena_floor):
            mario_y_vel[i] = mario_y_accel
        if(abs(mario_x_vel[i]) >= mario_max_x_vel):
            mario_x_vel[i] = xmod*mario_max_x_vel
        display_mario(mario_xs[id],mario_ys[id])
        pygame.display.update()
        clock.tick(fps)

def get_move(marios):
    return (random.randint(-1,2),random.randint(0,2))

def display_mario(x,y):
    global display
    pygame.draw.rect(display,(200,200,200),
            pygame.Rect(x,
                        arena_height - y - mario_height,
                        mario_width,
                        mario_height),0)


def main():
    print(mario_fight(np.array([2,4,5,1,8,34,67,90])))
    pygame.quit()

if __name__ == "__main__":
    main()
