from collections import namedtuple, deque
import numpy as np
from typing import Tuple, Set, Dict, List, Optional

# optimized data structure
GameState = namedtuple('GameState', [
    'board', # numpy array for visualization
    'snake_deque', # efficient  snake body tracking
    'snake_set', # 0(1) collision detection
    'green_apples', # set of positions
    'red_apple', # single position
    'size' # board size
    ])

def create_board(size: int = 10) -> GameState:
    """ creating initial game state"""
    #initializing empty board
    board =np.zeros((size, size), dtype=int)
    
    #placing snake in the middle
    x = size // 2
    y = size // 2
    snake_deque = deque([(x,y), (x, y-1),(x,y-2)])
    snake_set = set(snake_deque)

    # update board with snake
    for pos in snake_deque:
        board[pos] = 2 #snake body
    board[snake_deque[0]] = 1 # snake head

    #place apples
    green_apples = place_apples(board, snake_set, num_apples=2)
    red_apple = place_apples(board, snake_set.union(green_apples), num_apples=1).pop()

    return GameState(board, snake_deque, snake_set, green_apples, red_apple, size)


def place_apples(board: np.ndarray, occupied: Set[Tuple[int, int]], num_apples: int) -> set[Tuple[int, int]]:
    """ efficiently place apples in empty positions"""
    empty_positions = set((x, y) for x in range(board.shape[0]) 
                        for y in range(board.shape[1])) - occupied
    apple_positions = set()

    for _ in range(num_apples):
        pos = empty_positions.pop()
        apple_positions.add(pos)

    return apple_positions

def get_snake_vision(state: GameState) -> Dict[str, str]:
    """ get snake's  vision in all four directions"""
    head_x, head_y = state.snake_deque[0]
    directions = {
        'up': (-1, 0),
        'right': (0, 1),
        'down': (1, 0),
        'left': (0, -1)
    }
    vision = {}
    for direction, (dx, dy) in directions.items():
        vision[direction] = get_direction_vision(
            state, head_x, head_y, dx, dy
            )

    return vision


def get_direction_vision(
    state: GameState,
    x: int,
    y: int,
    dx: int,
    dy: int
) -> str:
    """ get vision in a single direction"""
    vision = []
    x, y = x + dx, y + dy

    while 0 <= x < state.size and 0 <= y < state.size:
        pos= (x,y)
        if pos in state.snake_set:
            vision.append('S')
        elif pos in state.green_apples:
            vision.append('G')
        elif pos == state.red_apple:
            vision.append('R')
        else:
            vision.append('0')
        x, y = x + dx, y + dy
    
    vision.append('W') # wall
    return ''.join(vision)


        
        
def move_snake(
    state: GameState,
    direction: str
 ) -> Tuple[GameState, bool, str]:
    """move snake and return new state, success flag, and message"""
    head_x, head_y = state.snake_deque[0]

    #calculate new head position
    direction_map = {
        'up':(-1, 0),
        'right':(0, 1),
        'down':(1, 0),
        'left':(0, -1)
    }
    dx, dy = direction_map[direction]
    new_head = (head_x + dx, head_y + dy)

    # wall collision
    if not (0 <= new_head[0] < state.size and 0 <= new_head[1] < state.size):
        return state, False, "wall_collision"

    # self collision
    if new_head in state.snake_set:
        return state, False, "self_collision"

    # create new snake state
    new_snake_deque = state.snake_deque.copy()
    new_snake_set = state.snake_set.copy()
    new_green_apples= state.green_apples.copy()
    new_red_apple = state.red_apple

    # handle apple collisions
    ate_apple = False
    if new_head in new_green_apples:
        ate_apple = True;
        new_green_apples.remove(new_head)
        new_green_apples.update(
            place_apples(state.board, 
                        state.snake_set.union({new_head}),
                        num_apples=1)
            )
    # handling the red apple collision
    elif new_head == new_red_apple:
        # check if snake length is already at minimum
        if len(new_snake_deque) <= 1:
            return state, False, "zero_length"
        #  remove tail properly (making sure it exists in set before removing)   
        old_tail= new_snake_deque.pop()
        if old_tail in new_snake_set: 
            new_snake_set.remove(old_tail)
        
        # placing new red apple
        new_red_apple = next(iter(
            place_apples(state.board,
                        new_snake_set.union(new_green_apples),
                        num_apples=1)
            ))

    # update snake position
    new_snake_deque.appendleft(new_head)
    new_snake_set.add(new_head)
    if not ate_apple:
        old_tail = new_snake_deque.pop()
        new_snake_set.remove(old_tail)

    # update board for visualization
    new_board = np.zeros_like(state.board)
    for pos in new_snake_deque:
        new_board[pos] = 2 # snake body
    new_board[new_head] = 1 # snake head
    for pos in new_green_apples:
        new_board[pos] = 3 # green apple
    if new_red_apple:
        new_board[new_red_apple] = 4 # red apple

    new_state = GameState(
        new_board,
        new_snake_deque,
        new_snake_set,
        new_green_apples,
        new_red_apple,
        state.size
    )
    
    return new_state, True, None 

def get_state_size(state: GameState) -> int:
    """ Get current snake length """
    return len(state.snake_deque)


