import pygame
import math
import numpy as np
import pickle
import neat
import os

# Initialize global variables
WIDTH = 900
DARK_BLUE = (58, 145, 181)
LIGHT_BLUE = (129, 198, 227)
DARK_GREEN = (64, 125, 88)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIGHT_GREEN = (96, 224, 147)
GREY = (128, 128, 128)
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Pathfinding NEAT Genetic Algorithm")

# Define class for each colored block
class Block:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.width = width
        
        # starting position of the drawn cubes
        self.x = row * width
        self.y = col * width
        
        # initialize all blocks to white
        self.color = WHITE
        self.font = pygame.font.Font('freesansbold.ttf', 10)
        self.score = ''
        
    def set_score(self, score):
        self.score = score
    
    def get_pos(self):
        return self.col, self.row
    
    def is_visited(self):
        return self.color == DARK_BLUE
    
    def is_checked(self):
        return self.color == LIGHT_BLUE
    
    def is_start(self):
        return self.color == LIGHT_GREEN

    def is_end(self):
        return self.color == DARK_GREEN
    
    def is_path(self):
        return self.color == RED

    def reset(self):
        self.color = WHITE
        
    def make_start(self):
        self.color = LIGHT_GREEN
        
    def make_visited(self):
        if not self.is_start() and not self.is_end():
            self.color = DARK_BLUE
        
    def make_checked(self):
        if not self.is_start() and not self.is_end():
            self.color = LIGHT_BLUE

    def make_end(self):
        self.color = DARK_GREEN
        
    def make_path(self):
        if not self.is_start() and not self.is_end():
            self.color = RED
        
    # draw the cube
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
        text_surface = self.font.render(self.score, True, WHITE, None)
        text_rect = (self.x + 0.4*(self.width), self.y + 0.4*(self.width))
        win.blit(text_surface, text_rect)

# initialize the grid  
def make_grid(num_rows, width):
    grid = []
    gap = width // num_rows
    for i in range(num_rows):
        grid.append([])
        for j in range(num_rows):
            block = Block(i, j, gap)
            grid[i].append(block)
    
    return grid

def draw_grid(win, num_rows, width):
    gap = width // num_rows
    for i in range(num_rows):
        # draw a horizontal line to separate every row
        pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
        
    for j in range(num_rows):
        # draw a vertical line to separate every column
        pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))

# draw the grids and each spots
def draw(win, grid, num_rows, width):
    win.fill(WHITE)
    
    for row in grid:
        for block in row:
            block.draw(win)
            
    draw_grid(win, num_rows, width)
    pygame.display.update()
    
# helper function to return row and col number from coordinates
def get_clicked_pos(pos, num_rows, width):
    gap = width // num_rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

# distance calculator
def dist(cur_x, cur_y, end_x, end_y):
    return math.sqrt((cur_x-end_x)**2 + (cur_y-end_y)**2)

def moveRight(cur_x, cur_y, grid):
    grid_width = len(grid[0])
    
    if cur_x + 1 < grid_width:
        nextBlock = grid[cur_x+1][cur_y]
        if not nextBlock.is_path():
            nextBlock.make_path()
            next_y, next_x = nextBlock.get_pos()
            return next_x, next_y, True
    return cur_x, cur_y, False
    
def moveLeft(cur_x, cur_y, grid):
    if cur_x - 1 >= 0:
        nextBlock = grid[cur_x-1][cur_y]
        if not nextBlock.is_path():
            nextBlock.make_path()
            next_y, next_x = nextBlock.get_pos()
            return next_x, next_y, True
    return cur_x, cur_y, False
    
def moveDown(cur_x, cur_y, grid):
    grid_height = len(grid)

    if cur_y + 1 < grid_height: 
        nextBlock = grid[cur_x][cur_y+1]
        if not nextBlock.is_path():
            nextBlock.make_path()
            next_y, next_x = nextBlock.get_pos()
            return next_x, next_y, True
    return cur_x, cur_y, False
    
def moveUp(cur_x, cur_y, grid):
    if cur_y - 1 >= 0:
        nextBlock = grid[cur_x][cur_y-1]
        if not nextBlock.is_path():
            nextBlock.make_path()
            next_y, next_x = nextBlock.get_pos()
            return next_x, next_y, True
    return cur_x, cur_y, False


def solveNN(draw, grid, start, end, num_rows, genomes, config):
    cur_y, cur_x = start.get_pos()
    end_y, end_x = end.get_pos()
    
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        
    clock = pygame.time.Clock()
    path_counter = 0
    while cur_x != end_x or cur_y != end_y:
        path_counter += 1
        clock.tick(30)
        # apply inputs to pre-trainedneural network
        output = net.activate((dist(cur_x, cur_y, end_x, end_y) - dist(cur_x, cur_y-1, end_x, end_y), 
                                               dist(cur_x, cur_y, end_x, end_y) - dist(cur_x, cur_y+1, end_x, end_y),
                                               dist(cur_x, cur_y, end_x, end_y) - dist(cur_x+1, cur_y, end_x, end_y),
                                               dist(cur_x, cur_y, end_x, end_y) - dist(cur_x-1, cur_y, end_x, end_y)))
        argmax = 0
        min_val = output[0]
        for i in range(len(output)):
            if output[i] < min_val:
                min_val = output[i]
                argmax = i
                
                
        valid = True
         # Up
        if argmax == 0:
            cur_x, cur_y, valid = moveUp(cur_x, cur_y, grid)
                
        # Down
        elif argmax == 1:
            cur_x, cur_y, valid = moveDown(cur_x, cur_y, grid)
                
        # Left
        elif argmax == 2:
            cur_x, cur_y, valid = moveLeft(cur_x, cur_y, grid)
                
        # Right
        elif argmax == 3:
            cur_x, cur_y, valid = moveRight(cur_x, cur_y, grid)
        
        draw()
        
        if not valid:
            print("Failed to solve")
            break
    
    if cur_x == end_x and cur_y == end_y:
        print("Path completed in", path_counter, "steps")

def main(win, width):
    pygame.init()
    num_rows = 10
    grid = make_grid(num_rows, width)
    
    # player decides start and end location
    start = None
    end = None
    
    run = True
    started = False
    finished = False
    
    while run:
        draw(win, grid, num_rows, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            # prevent user from triggering events when algorithm is underway
            if started and not finished:
                continue
            
            # [0] indicates left mouse button
            if not started: 
                if pygame.mouse.get_pressed()[0]:
                    # obtains mouse position
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, num_rows, width)
                    
                    # access the block object
                    block = grid[row][col]
                    
                    # if the "starting" block has not been initialized, set it first
                    if not start and block != end:
                        start = block
                        start.make_start()
                        
                    # if the "end" block has not been initialized, set it second
                    elif not end and block != start:
                        end = block
                        end.make_end()
                    
                # [2] indicates right mouse button to reset spots
                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_pos(pos, num_rows, width)
                    block = grid[row][col]
                    block.reset()
                    if block == start:
                        start = None
                    if block == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                # Click space to trigger the algorithm
                if event.key == pygame.K_SPACE and not started and start and end:
                    started = True
                    local_dir = os.path.dirname(__file__) # gives path to current directory
                    config_path = os.path.join(local_dir, "neat-config.txt")
                    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
                    
                    # load the pre-trained genome
                    with open('winner.pkl', 'rb') as f:
                        genome = pickle.load(f)
                    
                    genomes = [(1, genome)]
                    solveNN(lambda:draw(win, grid, num_rows, width), grid, start, end, num_rows, genomes, config)
                    finished = True
                
                # Click escape to restart the grid
                if event.key == pygame.K_ESCAPE and started:
                    # reset and empty grid
                    for i in range(num_rows):
                        for k in range(num_rows):
                            block = grid[i][k]
                            block.reset()
                            block.set_score('')
                    draw(win, grid, num_rows, width)
                    started = False
                    finished = False
                    start = None
                    end = None
    pygame.quit()
    
if __name__ == "__main__":
    main(WIN, WIDTH)
