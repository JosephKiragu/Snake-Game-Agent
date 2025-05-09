#!/usr/bin/env python3

import argparse
import os
import sys
import time
from typing import Tuple, Dict, Optional

from environment.board import create_board, move_snake, get_snake_vision, get_state_size
from agent.dqn import (
    init_training_state, encode_state, select_action, store_experience,
    optimize_model, save_model, load_model
)
from visualization.renderer import create_visualizer

# action mapping
ACTION_MAP = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}

# reward configs
REWARDS = {
    'green_apple': 10.0,    # positive reward for eating green apple
    'red_apple': -5.0,      # negative reward for eating red apple
    'empty': -0.01,         # small negative reward for moving without eating
    'wall_collision': -10.0, # larger negative reward for hitting wall
    'self_collision': -10.0, # larger negative reward for hitting self
    'zero_length': -10.0,    # larger negative reward for zero length
    'survival': 0.1,        # small positive reward for surviving each step
    'proximity_reward': 0.5 # reward for moving closer to green apple
}

class MetricsTracker:
    """performance tracking metrics"""
    def __init__(self):
        self.episode_lengths = []
        self.episode_rewards = []
        self.max_lengths = []
        self.durations = []

    def add_episode(self, length: int, reward: float, max_length: int, duration: int):
        self.episode_lengths.append(length)
        self.episode_rewards.append(reward)
        self.max_lengths.append(max_length)
        self.durations.append(duration)

    def get_summary(self, last_n: Optional[int] = None) -> Dict:
        if not self.episode_lengths:
            return {
                'avg_length': 0,
                'avg_reward': 0,
                'max_length': 0,
                'avg_duration': 0
            }

        if last_n is not None:
            lengths = self.episode_lengths[-last_n:]
            rewards = self.episode_rewards[-last_n:]
            max_lengths = self.max_lengths[-last_n:]
            durations = self.durations[-last_n:]
        else:
            lengths = self.episode_lengths
            rewards = self.episode_rewards
            max_lengths = self.max_lengths
            durations = self.durations

        return {
            'avg_length': sum(lengths) / len(lengths),
            'avg_reward': sum(rewards) / len(rewards),
            'max_length': max(max_lengths),
            'avg_duration': sum(durations) / len(durations)
        }


def calculate_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """ calculate manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def calculate_reward(
    state,
    previous_state,
    action: str,
    event: Optional[str]
) -> float:
    """calculate reward based on action and result"""
    reward = 0.0

    # event-based rewards
    if event == 'green_apple':
        reward += REWARDS['green_apple']
    elif event == 'red_apple':
        reward += REWARDS['red_apple']
    elif event == 'wall_collision':
        reward += REWARDS['wall_collision']
    elif event == 'self_collision':
        reward += REWARDS['self_collision']
    elif event == 'zero_length':
        reward += REWARDS['zero_length']
    else:
        reward += REWARDS['empty'] 

    # add survival reward
    reward += REWARDS['survival']

    # add proximity reward if applicable
    if previous_state and state:
        head = state.snake_deque[0]
        prev_head = previous_state.snake_deque[0]

        # find closest  green apple
        closest_apple= None
        min_distance = float('inf')
        for apple in state.green_apples:
            dist = calculate_distance(head, apple)
            if dist < min_distance:
                min_dist = dist
                closest_apple = apple
        if closest_apple:
            # calculate previous distance
            prev_dist = calculate_distance(prev_head, closest_apple)

            # if snake moved closer to apple, give proximity reward
            if min_dist < prev_dist:
                reward += REWARDS['proximity_reward']
    return reward


def run_episode(
    session: int,
    training_state: dict,
    visualizer=None,
    learn: bool = True,
    max_steps: int = 1000,
    metrics_tracker=None
) -> Tuple[int, float, int, int]:
    """Run a single episode of training"""
    state = create_board()
    vision = get_snake_vision(state)
    encoded_state = encode_state(vision)

    done = False
    step = 0
    total_reward = 0.0
    max_length = get_state_size(state)

    # get latest metrics for visualization
    current_metrics = metrics_tracker.get_summary(last_n=10) if metrics_tracker else None

    # visualization update
    if visualizer:
        visualizer.update(
            state, vision, "NONE", 0.0, session, step, current_metrics
        )
    while not done and step < max_steps:
        step +=1

        # select  action
        action_idx = select_action(encoded_state, training_state)
        action = ACTION_MAP[action_idx]

        # store previous state for reward calculation
        previous_state = state

        # execute action
        state, success, event = move_snake(state, action)

        # calculate reward
        reward = calculate_reward(state, previous_state, action, event)
        total_reward += reward

        # get new state
        if success:
            vision = get_snake_vision(state)
            next_encoded_state = encode_state(vision)
            done = False
        else:
            next_encoded_state = encoded_state #reuse last state for terminal
            done = True

        # store experience
        if learn:
            store_experience(
                training_state,
                encoded_state,
                action_idx,
                reward,
                next_encoded_state,
                done
            )

            # optimize model
            optimize_model(training_state)
        
        # update max length
        current_length = get_state_size(state)
        max_length = max(max_length, current_length)

        # visualize if requested
        if visualizer and (success or done):
            if not visualizer.update(
                state, vision, action, reward, session, step, current_metrics
            ):
                # user closed the window
                return step, total_reward, max_length, current_length
            
        # update state for next iteration
        encoded_state = next_encoded_state

    return step, total_reward, max_length, get_state_size(state)


def train(
    n_sessions: int,
    load_path: Optional[str] = None,
    save_path: Optional[str] = None,
    visual: bool = False,
    step_mode: bool = False,
    learn: bool = True,
    fps: int = 5,
    log_interval: int = 10
) -> MetricsTracker:
    """ run the full training process"""
    # initialize or load training state
    if load_path and os.path.exists(load_path):
        print(f"loading trained model from {load_path}")
        training_state = load_model(load_path)
    else:
        training_state = init_training_state()

    # initialize metrics
    metrics = MetricsTracker()

    # initialize visualizer if needed
    visualizer = None
    if visual:
        visualizer = create_visualizer(
            board_size=10,
            fps=fps,
            step_mode=step_mode
        )

    try:
        # run training sessions
        for session in range(1, n_sessions + 1):
            start_time = time.time()

            duration, reward, max_length, final_length = run_episode(
                session,
                training_state,
                visualizer=visualizer,
                learn=learn,
                metrics_tracker=metrics
            )

            # add to metrics
            metrics.add_episode(final_length, reward, max_length, duration)

            # log progress
            if session % log_interval == 0 or session == n_sessions:
                summary = metrics.get_summary(last_n=log_interval)
                print(f"session {session}/{n_sessions} - "
                      f"Duration: {duration}, Max Length: {max_length}, "
                      f"Avg Reward: {summary['avg_reward']:.2f}")

            # check if visualization was closed
            if visualizer and not visualizer.running:
                print("Visualization closed. Stopping training.")
                break

    finally:
        # clean up visualization resources
        if visualizer:
            visualizer.close()

    # save model if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_model(training_state, save_path)
        print(f"save training state in {save_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='snake reinforcement learning')
    parser.add_argument('-sessions',type=int, default=10, help='number of training sessions')
    parser.add_argument('-save', type=str, help='path to save model')
    parser.add_argument('-load', type=str, help='path to load the model')
    parser.add_argument('-visual', choices=['on', 'off'], default='off', help='enable visual display')
    parser.add_argument('-dontlearn', action='store_true', help='Disable learning during sessions')
    parser.add_argument('-step-by-step', action='store_true', help='Enable step-by-step mode')
    parser.add_argument('-fps', type=int, default=5, help='Frames per second for visualization')

    args = parser.parse_args()

    # create models directly if needed
    os.makedirs('models', exist_ok=True)

    # run training
    metrics = train(
        n_sessions=args.sessions,
        load_path=args.load,
        save_path=args.save,
        visual=args.visual == 'on',
        step_mode=args.step_by_step,
        learn=not args.dontlearn,
        fps=args.fps
    )

    # print final summary
    summary = metrics.get_summary()
    print("\nTraining complete!")
    print(f"game over, max length = {summary['max_length']}, max_duration = {int(summary['avg_duration'])}")


if __name__ == "__main__":
    main()
        
