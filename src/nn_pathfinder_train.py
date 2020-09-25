import pygame
import math
import numpy as np
import neat
import os
import random
import pickle
pygame.font.init()

# Initialize global variables
WIDTH = 900
DARK_BLUE = (58, 145, 181)
LIGHT_BLUE = (129, 198, 227)
DARK_GREEN = (64, 125, 88)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
LIGHT_GREEN = (96, 224, 147)
GREY = (206, 211, 219)
BLUE = (0, 89, 255)
STAT_FONT = pygame.font.SysFont("comicsans", 50)

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

    
    def get_pos(self):
        return self.col, self.row
    
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

    def make_end(self):
        self.color = DARK_GREEN
        
    def make_path(self):
        if not self.is_start() and not self.is_end():
            self.color = RED 
    
    # draw the cube
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

# initialize the grid  
def make_grid(num_rows, width, pop_size):
    pop_scale = int(math.sqrt(pop_size))
    grid = []
    gap = width // (num_rows*pop_scale)
    for i in range(num_rows):
        grid.append([])
        for j in range(num_rows):
            block = Block(i, j, gap)
            grid[i].append(block)
    return grid

def draw_grid(win, comb_rows, width, num_rows):
    gap = width // (comb_rows)
    for i in range(comb_rows):
        # draw a horizontal line to separate every row
        if i % num_rows == 0:
            pygame.draw.line(win, BLUE, (0, i*gap), (width, i*gap))
        else:
            pygame.draw.line(win, GREY, (0, i*gap), (width, i*gap))
        
    for j in range(comb_rows):
        # draw a vertical line to separate every column
        if j % num_rows == 0:
            pygame.draw.line(win, BLUE, (j*gap, 0), (j*gap, width))
        else:
            pygame.draw.line(win, GREY, (j*gap, 0), (j*gap, width))

# draw the grids and each spots
def draw(win, grid, comb_rows, width, num_rows, level):
    win.fill(WHITE)
    
    for i in range(comb_rows):
        for j in range(comb_rows):
            curBlock = grid[j][i]
            curBlock.draw(win)
            
    draw_grid(win, comb_rows, width, num_rows)
    text = STAT_FONT.render("Score: " + str(level), 1, BLACK)
    win.blit(text, (width - 10 - text.get_width(), 10))
    pygame.display.update()
    
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
            return next_x, next_y, 1
    return cur_x, cur_y, 0
    
def moveLeft(cur_x, cur_y, grid):
    if cur_x - 1 >= 0:
        nextBlock = grid[cur_x-1][cur_y]
        if not nextBlock.is_path():
            nextBlock.make_path()
            next_y, next_x = nextBlock.get_pos()
            return next_x, next_y, 1
    return cur_x, cur_y, 0
    
def moveDown(cur_x, cur_y, grid):
    grid_height = len(grid)

    if cur_y + 1 < grid_height: 
        nextBlock = grid[cur_x][cur_y+1]
        if not nextBlock.is_path():
            nextBlock.make_path()
            next_y, next_x = nextBlock.get_pos()
            return next_x, next_y, 1
    return cur_x, cur_y, 0
    
def moveUp(cur_x, cur_y, grid):
    if cur_y - 1 >= 0:
        nextBlock = grid[cur_x][cur_y-1]
        if not nextBlock.is_path():
            nextBlock.make_path()
            next_y, next_x = nextBlock.get_pos()
            return next_x, next_y, 1
    return cur_x, cur_y, 0

def combineGrids(grids, pop_size, width):
    # Note population count must be a SQUARE number
    comb_grid_count = int(math.sqrt(pop_size))
    grid_length = len(grids[0])
    gap = width // (grid_length*comb_grid_count)
    
    row_count = 0
    comb_symbols = []
    for i in range(comb_grid_count):
        inp_grid_segment = []
        for j in range(grid_length):
            inp_row = []
            for k in range(comb_grid_count):
                for z in range(grid_length):
                    curBlock = grids[k + row_count][j][z]
                    if curBlock.is_start():
                        inp_row.append("S")
                    elif curBlock.is_end():
                        inp_row.append("E")
                    elif curBlock.is_path():
                        inp_row.append("*")
                    else:
                        inp_row.append(".")
                    
            inp_grid_segment.append(inp_row)
        row_count += comb_grid_count
        comb_symbols.extend(inp_grid_segment)
    
    comb_list = []
    for i in range(len(comb_symbols)):
        comb_row = []
        for j in range(len(comb_symbols)):
            block = Block(i, j, gap)
            if comb_symbols[j][i] == 'S':
                block.make_start()
            elif comb_symbols[j][i] == 'E':
                block.make_end()
            elif comb_symbols[j][i] == '*':
                block.make_path()
            comb_row.append(block)
        comb_list.append(comb_row)
    return comb_list
        

def main(genomes, config):
    pygame.init()
    num_rows = 12
    nets = []
    ge = []
    
    # population size
    pop_size = 25
    comb_row_size = int(math.sqrt(pop_size))*num_rows
    valid_list = [True] * pop_size
    grids = [None] * pop_size
    cur_xs = [None] * pop_size
    cur_ys = [None] * pop_size
    iter_loop_lost = [False] * pop_size
    level_counter = 0
    
    # initialize the genomes
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)
    
    # create the grid and initialize start and end coordinates
    loop_counter = 0
    while not all(iter_loop_lost):
        loop_counter += 1
        if loop_counter >= 500:
            iter_loop_lost = [True] * pop_size
        iter_loop_finished = iter_loop_lost.copy()
        win = pygame.display.set_mode((WIDTH, WIDTH))
        pygame.display.set_caption("Pathfinding NEAT Genetic Algorithm")
        
        if not all(iter_loop_lost):
            level_counter += 1
        
        # reward model for surviving longer rounds
        for x, _ in enumerate(grids):
            if not iter_loop_finished[x]:
                ge[x].fitness += 5
        
        # determine the start and end coordinates 
        start_x = random.randint(0, num_rows-1)
        start_y = random.randint(0, num_rows-1)
        end_x = start_x
        end_y = start_y
        while start_x == end_x and start_y == end_y:
            end_x = random.randint(0, num_rows-1) 
            end_y = random.randint(0, num_rows-1)
        
        # reset grid parameters if it is still valid
        for j, _ in enumerate(grids):
            if valid_list[j]:      
                grid = make_grid(num_rows, WIDTH, pop_size)
                cur_x = start_x
                cur_y = start_y
                
                # draw the grid and initialize neural network input list
                for y in range(num_rows):
                    for x in range(num_rows):
                        curBlock = grid[x][y]
                        if x == start_x and y == start_y:
                            curBlock.make_start()
                        elif x == end_x and y == end_y:
                            curBlock.make_end()
                        else:
                            curBlock.reset()
                                
                grids[j] = grid
                cur_xs[j] = cur_x
                cur_ys[j] = cur_y
                iter_loop_finished[j] = False
        
        # combine all grids into one grid
        comb_grid = combineGrids(grids, pop_size, WIDTH)
        draw(win, comb_grid, comb_row_size, WIDTH, num_rows, level_counter)
        
        step_counter = 0
        clock = pygame.time.Clock()

        while not all(iter_loop_finished):
            clock.tick(90)  # n ticks per second
            # add an option to quit in the middle of the algorithm    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    
            for x, grid in enumerate(grids):
                if valid_list[x] and not iter_loop_finished[x]:
                    # punish model for very long paths
                    ge[x].fitness -= 0.2
                    
                    output = nets[x].activate((dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x], cur_ys[x]-1, end_x, end_y), 
                                               dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x], cur_ys[x]+1, end_x, end_y),
                                               dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x]+1, cur_ys[x], end_x, end_y),
                                               dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x]-1, cur_ys[x], end_x, end_y)))
                    
                    # find the argmax of the four directions
                    argmax = 0
                    min_val = output[0]
                    for w in range(len(output)):
                        if output[w] < min_val:
                            min_val = output[w]
                            argmax = w
                    
                    # Up
                    if argmax == 0:
                        cur_xs[x], cur_ys[x], valid = moveUp(cur_xs[x], cur_ys[x], grid)
                        # reward for moving closer to end
                        ge[x].fitness += (dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x], cur_ys[x]-1, end_x, end_y))/2
                        # print("Up")
                            
                    # Down
                    elif argmax == 1:
                        cur_xs[x], cur_ys[x], valid = moveDown(cur_xs[x], cur_ys[x], grid)
                        # reward for moving closer to end
                        ge[x].fitness += (dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x], cur_ys[x]+1, end_x, end_y))/2
                        # print("Down")
                            
                    # Left
                    elif argmax == 2:
                        cur_xs[x], cur_ys[x], valid = moveLeft(cur_xs[x], cur_ys[x], grid)
                        # reward for moving closer to end
                        ge[x].fitness += (dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x]-1, cur_ys[x], end_x, end_y))/2
                        # print("Left")
                            
                    # Right
                    elif argmax == 3:
                        cur_xs[x], cur_ys[x], valid = moveRight(cur_xs[x], cur_ys[x], grid)
                        # reward for moving closer to end
                        ge[x].fitness += (dist(cur_xs[x], cur_ys[x], end_x, end_y) - dist(cur_xs[x]+1, cur_ys[x], end_x, end_y))/2
                        # print("Right")
                        
                    if not valid:
                        # reduce fitness score if path crashes
                        ge[x].fitness = -1
                        valid_list[x] = False
                        iter_loop_finished[x] = True
                        iter_loop_lost[x] = True
                        break
                        
                    if cur_xs[x] == end_x and cur_ys[x] == end_y:
                        # increase fitness if end target reached
                        step_boost = 10
                        ge[x].fitness += step_boost
                        iter_loop_finished[x] = True
                        
            
            comb_grid = combineGrids(grids, pop_size, WIDTH)
            draw(win, comb_grid, comb_row_size, WIDTH, num_rows, level_counter)
            

    
    
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config) 
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main,10000) # number of generations
    pickle.dump(winner, open('winner.pkl', 'wb'))
    
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__) # gives path to current directory
    config_path = os.path.join(local_dir, "neat-config.txt")
    run(config_path)
