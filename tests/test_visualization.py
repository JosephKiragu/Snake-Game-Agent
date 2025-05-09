import pytest
import pygame
import numpy as np
from unittest.mock import MagicMock, patch
from collections import deque

from visualization.renderer import Visualizer, create_visualizer
from environment.board import create_board, get_snake_vision, GameState

class TestVisualizer:
    @pytest.fixture
    def mock_pygame(self, monkeypatch):
        """mock game to avoid actually initializing display for tests"""
        mock_display = MagicMock()
        mock_display.set_mode.return_value = MagicMock()
        mock_font = MagicMock()
        mock_font.SysFont.return_value = MagicMock()
        mock_font.SysFont.return_value.render.return_value = MagicMock()

        monkeypatch.setattr('pygame.display', mock_display)
        monkeypatch.setattr('pygame.font', mock_font)
        monkeypatch.setattr('pygame.draw', MagicMock())
        monkeypatch.setattr('pygame.init', MagicMock())
        monkeypatch.setattr('pygame.quit', MagicMock())

        # mock event system
        mock_event = MagicMock()
        mock_event.get.return_value = []
        monkeypatch.setattr('pygame.event', mock_event)

        return mock_display

    @pytest.fixture
    def mock_state(self):
        """create a mock game state"""
        state = create_board(size=10)
        return state

    @pytest.fixture
    def mock_vision(self):
        """create mock vision data"""
        return {
            'up': '0SW',
            'right': 'GS0W',
            'down': 'R0W',
            'left': '0SW'
        }

    def test_visualizer_init(self, mock_pygame):
        """test visualizer initialization"""
        vis = Visualizer(board_size=10)
        assert vis.board_size == 10
        assert vis.running == True
        assert vis.step_mode == False
        assert vis.frame_delay == 0
        mock_pygame.set_mode.assert_called_once()

    def test_set_speed(self, mock_pygame):
        """test setting visualization speed"""
        vis =   Visualizer()
        vis.set_speed(10)
        assert vis.frame_delay == 100 # 1000 / 10

        vis.set_speed(0)
        assert vis.frame_delay == 0 # no delay

    def test_step_mode(self, mock_pygame):
        """test setting step mode"""
        vis = Visualizer()
        assert vis.step_mode == False

        vis.set_step_mode(True)
        assert vis.step_mode == True

    def test_check_events_normal_mode(self, mock_pygame, monkeypatch):
        """test event handling in normal mode"""
        vis = Visualizer()

        # No events
        mock_event = MagicMock()
        mock_event.get.return_value = []
        monkeypatch.setattr('pygame.event', mock_event)

        assert vis.check_events() == True

        # Quit event
        mock_event.get.return_value = [MagicMock(type=pygame.QUIT)]
        assert vis.check_events() == False
        assert vis.running ==  False


    def test_update(self, mock_pygame, mock_state, mock_vision):
        """test visualization update"""
        vis = Visualizer()

        # mock rhe check events method to always return true
        vis.check_events = MagicMock(return_value=True)

        # test normal update
        result = vis.update(
            mock_state,
            mock_vision,
            "up",
            1.0,
            1,
            10,
            {"avg_reward": 0.5}
        )

        assert result == True
        assert vis.check_events.called

    def test_create_visualizer(self, mock_pygame):
        """test create_visualizer factory function"""
        with patch('visualization.renderer.Visualizer') as mock_vis_class:
            mock_vis = MagicMock()
            mock_vis_class.return_value = mock_vis

            vis = create_visualizer(
                board_size=15,
                fps=30,
                step_mode=True
            )

            mock_vis_class.assert_called_once_with(15)
            mock_vis.set_speed.assert_called_once_with(30)
            mock_vis.set_step_mode.assert_called_once_with(True)

    def test_draw_board(self, mock_pygame, mock_state):
        """test board drawing functionality"""
        vis = Visualizer()
        vis.draw_cell = MagicMock()

        vis.draw_board(mock_state)

        # should call draw_cell for every position
        assert vis.draw_cell.call_count > 0


def test_isolated_component():
    """test isolated component without pygame initialization"""
    with patch('pygame.init'):
        with patch('pygame.display.set_mode'):
            with patch('pygame.font.init'):
                with patch('pygame.font.SysFont') as mock_font:
                    mock_font.return_value = MagicMock()
                    vis = create_visualizer(10, 5, False)
                    assert vis is not None
    
