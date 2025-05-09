import pytest
import torch
import numpy as np
from agent.dqn import (
    create_dqn,
    init_training_state,
    encode_state,
    select_action,
    store_experience,
    optimize_model,
    save_model,
    load_model
)

@pytest.fixture
def mock_vision():
    return {
        'up':'0SW',
        'right':'GS0W',
        'down':'R0W',
        'left':'0SW'
    }

@pytest.fixture
def training_state():
    return init_training_state()


def test_create_dqn():
    model = create_dqn()
    assert isinstance(model, torch.nn.Module)
    assert len(list(model.parameters())) > 0

    # test forward pass
    test_input = torch.randn(1, 16)
    output = model(test_input)
    assert output.shape == (1, 4)


def test_init_training_state():
    state = init_training_state()
    required_keys= ['device', 'policy_net', 'target_net', 'optimizer',
                    'memory', 'epsilon', 'steps']

    assert all(key in state for key in required_keys)
    assert isinstance(state['policy_net'], torch.nn.Module)
    assert isinstance(state['target_net'], torch.nn.Module)
    assert state['epsilon'] == 1.0
    assert state['steps'] == 0

def test_encode_state(mock_vision):
    encoded = encode_state(mock_vision)
    assert isinstance(encoded, torch.FloatTensor)
    assert encoded.shape == (16,) # 4 directions * 4 features
    assert torch.all(encoded >= 0) and torch.all(encoded <= 1)


def test_select_action(training_state, mock_vision):
    state = encode_state(mock_vision)
    action = select_action(state, training_state)
    assert isinstance(action, int)
    assert 0 <= action <= 3


def test_store_experience(training_state, mock_vision):
    state = encode_state(mock_vision)
    next_state = encode_state(mock_vision)

    initial_memory_size = len(training_state['memory'])
    store_experience(training_state, state, 0, 1.0, next_state, False)

    assert len(training_state['memory']) == initial_memory_size + 1
    experience = training_state['memory'][-1]
    assert torch.equal(experience.state, state)
    assert experience.action == 0
    assert experience.reward == 1.0
    assert torch.equal(experience.next_state, next_state)
    assert experience.done is False

def test_optimize_model(training_state, mock_vision):
    # fill memory with some experience
    state = encode_state(mock_vision)
    next_state = encode_state(mock_vision)

    for _ in range(128): # Batch size
        store_experience(training_state, state, 0, 1.0, next_state, False)

    initial_steps = training_state['steps']
    initial_epsilon = training_state['epsilon']

    optimize_model(training_state)

    assert training_state['steps'] == initial_steps + 1
    assert training_state['epsilon'] < initial_epsilon


@pytest.mark.parametrize("test_dir", ["test_models"])
def test_save_load_model(training_state, test_dir, tmp_path):
    save_path = tmp_path / f"{test_dir}/model.pt"
    save_path.parent.mkdir(exist_ok="True")

    # save_model
    save_model(training_state, str(save_path))
    assert save_path.exists()

    # load model
    loaded_state = load_model(str(save_path))

    # verify loaded state matches the original
    assert loaded_state['epsilon'] == training_state['epsilon']
    assert loaded_state['steps'] == training_state['steps']


    # verify models have the same parameters
    for orig_param, loaded_param in zip(
        training_state['policy_net'].parameters(),
        loaded_state['policy_net'].parameters()
    ):
        assert torch.equal(orig_param.data, loaded_param.data)



def test_full_training_cycle(training_state, mock_vision):
    # simulate multiple training steps
    state = encode_state(mock_vision)

    for _ in range(200):
        action = select_action(state, training_state)
        next_state = encode_state(mock_vision) # in real scenario, this would be from environment
        reward = 1.0 # mock reward
        done = False

        store_experience(training_state, state, action, reward, next_state, done)
        optimize_model(training_state)

        state = next_state

        if done:
            break
    
    assert training_state['steps'] > 0
    assert training_state['epsilon'] < 1.0

