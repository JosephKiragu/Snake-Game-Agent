import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
TARGET_UPDATE = 10

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    def __init__(self, input_size=16):  # 4 directions * 4 state variables
        super(DuelingDQN, self).__init__()
        
        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 actions: UP, RIGHT, DOWN, LEFT
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling architecture formula
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

def create_dqn() -> nn.Module:
    """Create the DQN model with the dueling architecture"""
    return DuelingDQN()

def init_training_state():
    """Initialize training state with both networks and optimizer"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = create_dqn().to(device)
    target_net = create_dqn().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    
    return {
        'device': device,
        'policy_net': policy_net,
        'target_net': target_net,
        'optimizer': optimizer,
        'memory': memory,
        'epsilon': epsilon,
        'steps': 0
    }

def encode_state(vision: Dict[str, str]) -> torch.Tensor:
    """Convert snake's vision into a tensor for the neural network"""
    # Create encoding for each direction
    state_vector = []
    for direction in ['up', 'right', 'down', 'left']:
        view = vision[direction]
        # Encode each position type
        wall = 1.0 if 'W' in view else 0.0
        snake = 1.0 if 'S' in view else 0.0
        green = 1.0 if 'G' in view else 0.0
        red = 1.0 if 'R' in view else 0.0
        state_vector.extend([wall, snake, green, red])
    
    return torch.FloatTensor(state_vector)

def select_action(state: torch.Tensor, training_state: dict) -> int:
    """Select action using epsilon-greedy strategy"""
    if random.random() < training_state['epsilon']:
        return random.randint(0, 3)
    
    with torch.no_grad():
        state = state.unsqueeze(0).to(training_state['device'])
        q_values = training_state['policy_net'](state)
        return q_values.max(1)[1].item()

def store_experience(
    training_state: dict,
    state: torch.Tensor,
    action: int,
    reward: float,
    next_state: torch.Tensor,
    done: bool
):
    """Store experience in replay memory"""
    training_state['memory'].append(
        Experience(state, action, reward, next_state, done)
    )

def optimize_model(training_state: dict):
    """Perform one step of optimization on the DQN"""
    if len(training_state['memory']) < BATCH_SIZE:
        return
    
    device = training_state['device']
    experiences = random.sample(training_state['memory'], BATCH_SIZE)
    
    # Transpose batch of experiences
    batch = Experience(*zip(*experiences))
    
    # Convert to tensors
    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.tensor(batch.reward, device=device)
    next_state_batch = torch.stack(batch.next_state).to(device)
    done_batch = torch.tensor(batch.done, device=device)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = training_state['policy_net'](state_batch).gather(1, action_batch.unsqueeze(1))
    
    # Compute V(s_{t+1}) for all next states
    with torch.no_grad():
        next_state_values = training_state['target_net'](next_state_batch).max(1)[0]
        next_state_values[done_batch] = 0.0  # Set to 0 for terminal states
    
    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)
    
    # Compute Huber loss
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    training_state['optimizer'].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(training_state['policy_net'].parameters(), 100)
    training_state['optimizer'].step()
    
    # Update epsilon
    training_state['epsilon'] = max(
        EPSILON_END,
        training_state['epsilon'] * EPSILON_DECAY
    )
    
    # Update target network
    training_state['steps'] += 1
    if training_state['steps'] % TARGET_UPDATE == 0:
        training_state['target_net'].load_state_dict(
            training_state['policy_net'].state_dict()
        )

def save_model(training_state: dict, path: str):
    """Save the DQN model and training state"""
    torch.save({
        'policy_net_state_dict': training_state['policy_net'].state_dict(),
        'target_net_state_dict': training_state['target_net'].state_dict(),
        'optimizer_state_dict': training_state['optimizer'].state_dict(),
        'epsilon': training_state['epsilon'],
        'steps': training_state['steps']
    }, path)

def load_model(path: str) -> dict:
    """Load a saved DQN model and training state"""
    training_state = init_training_state()
    checkpoint = torch.load(path)
    
    training_state['policy_net'].load_state_dict(checkpoint['policy_net_state_dict'])
    training_state['target_net'].load_state_dict(checkpoint['target_net_state_dict'])
    training_state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
    training_state['epsilon'] = checkpoint['epsilon']
    training_state['steps'] = checkpoint['steps']
    
    return training_state