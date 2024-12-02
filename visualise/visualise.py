"""
VISUALISATION
"""

import sys
import random
import pygame

def dist_visualise(width: int, height: int, start: tuple[int, int], end: tuple[int, int],\
                   path: list[tuple[int, int]]):
    """
    Visualises Manhattan distance on matrix.

    Keyword arguments:
    ------------------
    width & height:
      int, the size of matrix (width x height)
    start & end:
      tuple of ints, the first and the last coordinates (x;y)
    path:
      list of int tuples, list of coordinates (x;y), which
      are supposed to connect the strat and the end.
    """
    screenw = 2400
    screenh = 750
    parameter = min(screenw//width, screenh//height)

    screen = pygame.display.set_mode((parameter * width, parameter * height))
    pygame.display.set_caption("Distance visualisation")
    pictures = ['lisa.png','dutchgirl.png','murashko.png','frenchgirl.png','kasiyan.png']
    backposes = []

    xcount = 1
    ycount = 1
    for _ in range(height):
        for _ in range(width):
            backposes.append((xcount, ycount))
            xcount += parameter
        xcount = 1
        ycount += parameter
        backposes.append((xcount, ycount))

    startpoint = pygame.image.load("first.png")
    startpoint = pygame.transform.scale(startpoint, (parameter, parameter))

    finpoint = pygame.image.load("last.png")
    finpoint = pygame.transform.scale(finpoint, (parameter, parameter))
    goals = [((start[0] - 1) * parameter, (start[1] - 1) * parameter),\
                         ((end[0] - 1) * parameter, (end[1] - 1) * parameter)]

    pathframe = pygame.image.load("path.png")
    pathframe = pygame.transform.scale(pathframe, (parameter, parameter))

    def launch():
        """
        launches the programm
        """
        path_index = 0
        clock = pygame.time.Clock()
        delay = 1000
        stop = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if stop == 0:
                for pos in backposes:
                    screen.blit(pygame.transform.scale(\
                        pygame.image.load(random.choice(pictures)), (parameter, parameter)), pos)
                stop += 1
            screen.blit(startpoint, goals[0])
            screen.blit(finpoint, goals[1])
            for i in range(path_index + 1):
                if i < len(path):
                    path_square_pos = ((path[i][0] - 1) * int(parameter),\
                                        (path[i][1] - 1) * int(parameter))
                    screen.blit(pathframe, path_square_pos)

            pygame.display.flip()
            if path_index < len(path) - 1:
                path_index += 1
                pygame.time.delay(delay)

            clock.tick(40)
    launch()

dist_visualise(20, 20, (3, 4), (17, 18), [
    (4, 4), (4, 5), (5, 5), (5, 6), (6, 6), (6, 7), (7, 7), (7, 8), 
    (8, 8), (8, 9), (9, 9), (9, 10), (10, 10), (10, 11), (11, 11), 
    (11, 12), (12, 12), (12, 13), (13, 13), (13, 14), (14, 14), (14, 15), 
    (15, 15), (15, 16), (16, 16), (16, 17), (17, 17)
])
