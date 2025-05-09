import pygame
import numpy as np
import time
from typing import Dict, Tuple, Optional

# constants for visualization
CELL_SIZE = 40
GRID_COLOR = (50, 50, 50)
BG_COLOR = (0, 0, 0)
COLORS = {
    'head': (30, 144, 255), # blue
    'body': (0, 100, 200), # darker blue
    'green': (0, 255, 0), # green
    'red': (255, 0, 0), # red
    'text': (255, 255, 255) # white
}

class Visualizer:
    def __init__(self, board_size: int = 10):
        """ initialize pygame and set up display"""
        pygame.display.init()
        pygame.font.init()

        self.board_size = board_size
        self.screen_width = self.board_size * CELL_SIZE + 400 # extra space for info panel
        self.screen_height = self.board_size * CELL_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Snake Reinforcement Learning")
        self.font = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()
        self.frame_delay = 0 # no delay initially
        self.step_mode = False
        self.running = True

    def set_speed(self, fps: int):
        """set visualization speed """
        self.frame_delay = 1000 // fps if fps > 0 else 0

    def set_step_mode(self, enabled: bool):
        """ enable or disable step-by-step mode"""
        self.step_mode = enabled

    def check_events(self) -> bool:
        """process pygame ecents, return whether to continye or wait for next step"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                if event.key == pygame.K_SPACE and self.step_mode:
                    return True
                # speed controls
                if event.key == pygame.K_UP:
                    self.frame_delay = max(0, self.frame_delay - 50)
                if event.key == pygame.K_DOWN:
                    self.frame_delay += 50
        return not self.step_mode

    def draw_cell(self, x: int, y: int, color: Tuple[int, int, int]):
        """Draw a single cell at the given coordinates"""
        rect = pygame.Rect(
            x * CELL_SIZE,
            y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

    def draw_board(self, state):
        """draw the board with snake and apples"""
        self.screen.fill(BG_COLOR)

        # Draw grid
        for y in range(self.board_size):
            for x in range(self.board_size):
                self.draw_cell(x, y, BG_COLOR)

        # draw snake body
        for segment in list(state.snake_deque)[1:]:
            self.draw_cell(segment[1],segment[0], COLORS['body'])

        # draw snake head
        if state.snake_deque:
            head = state.snake_deque[0]
            self.draw_cell(head[1], head[0], COLORS['head'])

        # draw green apples
        for apple in state.green_apples:
            self.draw_cell(apple[1], apple[0], COLORS['green'])

        # draw red apple
        if state.red_apple:
            self.draw_cell(state.red_apple[1], state.red_apple[0], COLORS['red'])

    def draw_info_panel(
        self,
        vision: Dict[str, str],
        action: str,
        reward: float,
        length: int,
        episode: int,
        steps: int,
        metrics: Optional[Dict] = None
    ):

        """draw information panel with game state and metrics"""
        panel_x = self.board_size * CELL_SIZE + 10
        y_offset = 20

        # episode info
        self.draw_text(f"Episode: {episode}", panel_x, y_offset)
        y_offset += 30
        self.draw_text(f"Steps: {steps}", panel_x, y_offset)
        y_offset += 30
        self.draw_text(f"Snake Length: {length}", panel_x, y_offset)
        y_offset+=30

        # action and reward
        self.draw_text(f"Action: {action}", panel_x, y_offset)
        y_offset += 30
        self.draw_text(f"Reward: {reward:.2f}", panel_x, y_offset)
        y_offset += 30

        # vision
        self.draw_text("snake vision:", panel_x, y_offset)
        y_offset += 30
        for direction, view in vision.items():
            self.draw_text(f"{direction.upper()}: {view}", panel_x, y_offset)
            y_offset += 25
        y_offset += 10

        # metrics if available
        if metrics:
            self.draw_text("performance metrics:", panel_x, y_offset)
            y_offset += 30
            for key, value in metrics.items():
                self.draw_text(f"{key}: {value:.2f}", panel_x, y_offset)
                y_offset += 25

        # controls 
        y_offset = self.screen_height - 150 
        self.draw_text("controls:", panel_x, y_offset)
        y_offset += 30
        self.draw_text("ESC - Quit ", panel_x, y_offset)
        y_offset += 25
        self.draw_text("SPACE - Next step (step mode)", panel_x, y_offset)
        y_offset += 25
        self.draw_text("UP/DOWN - Adjust speed", panel_x, y_offset)
        y_offset += 25
        self.draw_text(f"Current delay: {self.frame_delay}ms", panel_x, y_offset)


    def draw_text(self, text: str, x: int, y: int):
        """ render text on screen"""
        text_surface = self.font.render(text, True, COLORS['text'])
        self.screen.blit(text_surface, (x, y))

    def update(
        self,
        state,
        vision: Dict[str, str],
        action: str,
        reward: float,
        episode: int,
        steps: int,
        metrics: Optional[Dict] = None
    ) -> bool:
        """ update the display and wait for next step in step mode"""
        self.draw_board(state)
        self.draw_info_panel(
            vision,
            action,
            reward,
            len(state.snake_deque),
            episode,
            steps,
            metrics
        )

        pygame.display.flip()

        if self.frame_delay > 0:
            time.sleep(self.frame_delay / 1000)

        # in step mode wait for space key
        if self.step_mode:
            waiting = True
            while waiting and self.running:
                waiting = not self.check_events()
                self.clock.tick(30)
        else:
            self.check_events()
            self.clock.tick(60)

        return self.running

    def close(self):
        """ clean up pygame resources"""
        pygame.quit()



# intergration function for display in training pipeline
def create_visualizer(board_size: int = 10, fps: int = 5, step_mode: bool = False) -> Visualizer:
    """ create and configure the visualizer"""
    vis = Visualizer(board_size)
    vis.set_speed(fps)
    vis.set_step_mode(step_mode)
    return vis
