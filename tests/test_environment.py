import pytest
import numpy as np
from collections import deque
from environment.board import (
    create_board,
    move_snake,
    get_snake_vision,
    get_state_size
)

def test_board_creation():
    state = create_board(size=10)
    assert state.size == 10
    assert state.board.shape == (10, 10)
    assert len(state.snake_deque) == 3
    assert len(state.green_apples) == 2
    assert state.red_apple is not None

def test_snake_movement():
    state = create_board()
    initial_length = len(state.snake_deque)

    # test valid move
    new_state, success, msg = move_snake(state, 'right')
    assert success
    assert len(new_state.snake_deque) == initial_length

    # test_wall collision
    for  _ in range(10):
        state, success, msg = move_snake(state, 'right')
    assert not success
    assert msg == "wall_collision"

def test_apple_collison():
    state = create_board()

    # place green apple in front of snake
    head = state.snake_deque[0]
    state = state._replace(
        green_apples={tuple(map(sum, zip(head, (0,1))))},
        board=np.zeros((10,10))
    )

    new_state, success, msg = move_snake(state, 'right')
    assert success
    assert len(new_state.snake_deque) == len(state.snake_deque) + 1

def test_snake_vision():
    state = create_board()
    vision = get_snake_vision(state)

    assert all(direction in vision for direction in ['up', 'down', 'left', 'right'])
    assert all('W' in value for value in vision.values()) # all direction see walls

def test_self_collison():
    state = create_board();
    moves = ['right', 'down', 'left', 'up']

    for move in moves:
        state, success, msg = move_snake(state, move)

    assert not success
    assert msg == "self_collision" 

def test_red_apple_death():
    state = create_board()
    # place red apple in front of length snake 1
    state = state._replace(
        snake_deque=deque([state.snake_deque[0]]),
        snake_set={state.snake_deque[0]},
        red_apple=tuple(map(sum, zip(state.snake_deque[0], (0,1))))
    )

    new_state, success, msg = move_snake(state, 'right')
    assert not success
    assert msg == "zero_length"
