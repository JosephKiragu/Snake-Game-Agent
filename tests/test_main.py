import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from main import (
    calculate_reward,
    calculate_distance,
    run_episode,
    train,
    MetricsTracker
)

def test_calculate_distance():
    """test manhattan distance calculation"""
    assert calculate_distance((0,0), (3,4)) == 7
    assert calculate_distance((2,3), (5,7)) == 7
    assert calculate_distance((1,1), (1,1)) == 0


def test_calculate_reward():
    """test reward calculation logic"""

    # mock states and test various reward scenarios
    mock_state = MagicMock()
    mock_state.snake_deque=[(5,5)]
    mock_state.green_apples= {(6,5)}

    mock_prev_state = MagicMock()
    mock_prev_state.snake_deque = [(4,5)]

    # test green apple reward
    reward = calculate_reward(mock_state, mock_prev_state, 'right', 'green_apple')
    assert reward > 0

    # test collision penalty
    reward = calculate_reward(mock_state, mock_prev_state, 'right', 'wall_collision')
    assert reward < 0

@pytest.fixture
def mock_training_components():
    """mock components needed for training"""
    with patch('main.create_board') as mock_create_board, \
         patch('main.move_snake') as mock_move_snake, \
         patch('main.get_snake_vision') as mock_get_vision, \
         patch('main.encode_state') as mock_encode, \
         patch('main.select_action') as mock_select, \
         patch('main.store_experience') as mock_store, \
         patch('main.optimize_model')as mock_optimize:

        
        # setup mock returns
        mock_state = MagicMock()
        mock_state.snake_deque = [(5,5), (5,4), (5,3)]
        mock_create_board.return_value = mock_state

        mock_vision = {'up': '0W', 'right': 'GW', 'down': '0W', 'left': '0W'}
        mock_get_vision.return_value = mock_vision
        
        mock_encoded = torch.zeros(16)
        mock_encode.return_value = mock_encoded

        mock_select.return_value = 1 #right
        
        mock_move_snake.side_effect = [(mock_state, True, None), (mock_state, False, 'wall_collision')]

        yield {
            'state': mock_state,
            'vision': mock_vision,
            'encoded': mock_encoded
        }


def test_run_episode(mock_training_components):
    """test single episode execution"""
    mock_training_state = {
        'policy_net': MagicMock(),
        'target_net': MagicMock(),
        'optimizer': MagicMock(),
        'memory': [],
        'epsilon': 0.1,
        'steps': 0,
        'device': 'cpu'
    }

    # test with learning enabled
    with patch('main.get_state_size', return_value=3):
        duration, reward, max_length, final_length = run_episode(
            1, mock_training_state, visualizer=None, learn=True, max_steps=10
        )

    # should end after second step (wall collision)
    assert duration == 2
    assert max_length == 3


def test_metrics_tracker():
    """test metrics tracking functionality"""
    tracker = MetricsTracker()

    # add some episodes
    tracker.add_episode(3, 10.5, 3, 20)
    tracker.add_episode(4, 15.2, 4,30)
    
    # get summary
    summary = tracker.get_summary()
    assert summary['avg_length'] == 3.5
    assert summary['avg_reward'] == 12.85
    assert summary['max_length'] == 4
    assert summary['avg_duration'] == 25

    # test last_n functionality
    tracker.add_episode(5, 20.0, 5, 40)
    last_summary = tracker.get_summary(last_n=1)
    assert last_summary['avg_length'] == 5
