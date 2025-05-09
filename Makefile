.PHONY: all clean test test-file lint setup train run models

PYTHON=python3
VENV=venv
BIN=$(VENV)/bin
MODEL_DIR=models

all: setup test lint

setup: 
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

test: 
	PYTHONPATH=src $(BIN)/pytest tests/ -v

test-file: 
	PYTHONPATH=src $(BIN)/pytest tests/$(word 2,$(MAKECMDGOALS)) -v

%:
	@:


lint:
	$(BIN)/flake8 snake_rl/
	$(BIN)/mypy snake_rl/

# training tasks
train-1:
	PYTHON_PATH=. $(BIN)/python -m src.main -sessions 1 -save $(MODEL_DIR)/1sess.txt -visual off

train-10:
	PYTHON_PATH=. $(BIN)/python -m src.main -sessions 10 -save $(MODEL_DIR)/10sess.txt -visual off

train-100:
	PYTHON_PATH=. $(BIN)/python -m src.main -sessions 100 -save $(MODEL_DIR)/100sess.txt -visual off

train-1000:
	PYTHON_PATH=. $(BIN)/python -m src.main -sessions 1000 -save $(MODEL_DIR)/1000sess.txt -visual off

# run with visualization
run -1:
	PYTHON_PATH=. $(BIN)/python -m src.main -load $(MODEL_DIR)/1sess.txt -visual on -dontlearn

run-10:
	PYTHON_PATH=. $(BIN)/python -m src.main -load $(MODEL_DIR)/10sess.txt -visual on -dontlearn

run-100:
	PYTHON_PATH=. $(BIN)/python -m src.main -load $(MODEL_DIR)/100sess.txt -visual on -dontlearn

run-1000:
	PYTHON_PATH=. $(BIN)/python -m src.main -load $(MODEL_DIR)/1000sess.txt -visual on -dontlearn

# run in step-by-step mode
debug-run:
	PYTHON_PATH=. $(BIN)/python -m src.main -load $(MODEL_DIR)/100sess.txt -visual on -dontlearn -step-by-step

# generate all required models
models: train-1 train-10 train-100

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete

