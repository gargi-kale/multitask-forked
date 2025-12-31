# Makefile for setting up and managing the project

# Configuration
VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python3
PIP=${VENV_NAME}/bin/pip3

# Default target executed when no arguments are given to make.
default: setup

# Set up python interpreter environment
.PHONY: setup
setup: 
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt


# Run the project with custom config using the text
.PHONY: run
run: 
ifdef config_path
	@echo "Running the project with custom configuration using the text; computing the embeddings..."
	$(PYTHON) main.py $(config_path)
else
	@echo "Please provide a config path. Usage: make run config_path=<path_to_config>"
endif


# Run the project with custom config using the embeddings
.PHONY: run_embedding
run: 
ifdef config_path
	@echo "Running the project with custom configuration using precomputed embeddings..."
	$(PYTHON) main_embedding.py $(config_path)
else
	@echo "Please provide a config path. Usage: make run config_path=<path_to_config>"
endif


# Run the project multiple times with different configs
.PHONY: multirun
multirun: 
ifdef config_paths
	$(foreach config,$(config_paths), \
		@echo "Running the project with configuration $(config)..."; \
		$(PYTHON) main.py $(config); \
	)
else
	@echo "Please provide config paths. Usage: make multirun config_paths=\"<path1> <path2> ...\""
endif



# Run the project multiple times with different configs
.PHONY: multirun_embedding
multirun: 
ifdef config_paths
	$(foreach config,$(config_paths), \
		@echo "Running the project with configuration $(config)..."; \
		$(PYTHON) main_embedding.py $(config); \
	)
else
	@echo "Please provide config paths. Usage: make multirun config_paths=\"<path1> <path2> ...\""
endif



# Clean up the project
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	find . -type d -name '__pycache__' -delete

# Help
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "setup       		- Set up the virtual environment and install dependencies"
	@echo "run         		- Run the main project script with a custom configuration file using the text, computing the embeddings"
	@echo "run_embedding 	- Run the main project script with a custom configuration file using precomputed embeddings"
	@echo "multirun  		- Run the project multiple times with different configuration files"
	@echo "clean       		- Clean up the project (remove virtual environment and temporary files)"
	@echo "help        		- Display this help message"
