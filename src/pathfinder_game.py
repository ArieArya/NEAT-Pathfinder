import pygame
import math
import numpy as np

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
pygame.display.set_caption("Pathfinding with A* Algorithm")

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
    
    def is_barrier(self):
        return self.color == BLACK
    
    def is_start(self):
        return self.color == LIGHT_GREEN

    def is_end(self):
        return self.color == DARK_GREEN

    def reset(self):
        self.color = WHITE
        
    def make_start(self):
        self.color = LIGHT_GREEN
    
    def make_barrier(self):
        self.color = BLACK

    def make_end(self):
        self.color = DARK_GREEN
        
    def make_path(self):
        if not self.is_start() and not self.is_end():
            self.color = RED
        
    # draw the cube
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

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

def moveRight(cur, grid):
    grid_width = len(grid[0])
    cur_y, cur_x = cur.get_pos()
    
    if cur_x + 1 < grid_width:
        nextBlock = grid[cur_x+1][cur_y]
        nextBlock.make_path()
        return nextBlock
    return cur
    
def moveLeft(cur, grid):
    cur_y, cur_x = cur.get_pos()

    if cur_x - 1 >= 0:
        nextBlock = grid[cur_x-1][cur_y]
        nextBlock.make_path()
        return nextBlock
    return cur
    
def moveDown(cur, grid):
    grid_height = len(grid)
    cur_y, cur_x = cur.get_pos()

    if cur_y + 1 < grid_height:
        nextBlock = grid[cur_x][cur_y+1]
        nextBlock.make_path()
        return nextBlock
    return cur
    
def moveUp(cur, grid):
    cur_y, cur_x = cur.get_pos()

    if cur_y - 1 >= 0:
        nextBlock = grid[cur_x][cur_y-1]
        nextBlock.make_path()
        return nextBlock
    return cur
    

# --------------------- NEURAL NETWORK SECTION -----------------------------
def solveNN(draw, grid, start, end, num_rows):
    cur = start
    step_counter = 0

    while cur != end:
        # add an option to quit in the middle of the algorithm
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    cur = moveUp(cur, grid)
                    step_counter += 1
                elif event.key == pygame.K_DOWN:
                    cur = moveDown(cur, grid)
                    step_counter += 1
                elif event.key == pygame.K_RIGHT:
                    cur = moveRight(cur, grid)
                    step_counter += 1
                elif event.key == pygame.K_LEFT:
                    cur = moveLeft(cur, grid)
                    step_counter += 1
        draw()
    print("Completed in ", step_counter, " steps")
        
        
def main(win, width):
    pygame.init()
    num_rows = 30
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
                        
                    # next, just initialize all the barriers
                    elif block != start and block != end:
                        block.make_barrier()
                    
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
                    solveNN(lambda:draw(win, grid, num_rows, width), grid, start, end, num_rows)
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
