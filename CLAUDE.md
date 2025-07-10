# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project implementing backpropagation neural networks using MXNet tensors for optimal performance. The project includes both Python (main implementation) and Perl versions, with a focus on converting traditional loop-based neural network operations to vectorized tensor operations.

## Development Environment

### Docker Setup (Recommended)
```bash
# Quick start
./run.sh

# Manual Docker commands
docker-compose build
docker-compose up -d
docker-compose logs ml-notebook | grep token
```

### Local Development
```bash
# Install dependencies
pip install mxnet==1.9.1 plotly==5.17.0 jupyter notebook pandas numpy

# For Perl implementation
# Requires AI::MXNet, Chart::Plotly modules
```

## Architecture

### Core Implementation Files
- **`sml.py`**: Main Python module containing all tensor-based neural network functions
- **`sml.pm`**: Perl module with equivalent functionality
- **`Chapter15-Backpropagation-Tensores-Python.ipynb`**: Interactive demonstration notebook

### Key Tensor Functions (sml.py)
- `activate(weights, inputs)`: Tensor-based neuron activation using mx.nd.dot
- `transfer(activation)`: Vectorized sigmoid function
- `initialize_network(n_inputs, n_hidden, n_outputs)`: Creates network with MXNet tensors
- `forward_propagate(network, row)`: Vectorized forward pass
- `backward_propagate_error(network, expected)`: Tensor-based backpropagation
- `update_weights(network, row, l_rate)`: Vectorized weight updates
- `train_network()`: Complete training loop with tensor operations
- `back_propagation()`: Full algorithm implementation

### Data Processing Functions
- `load_csv()`, `str_column_to_float()`, `str_column_to_int()`: Data loading utilities
- `dataset_minmax()`, `normalize_dataset()`: Preprocessing with tensor support
- `train_test_split()`, `cross_validation_split()`: Data splitting utilities
- `accuracy_metric()`, `rmse_metric()`: Evaluation metrics

## Development Commands

### Testing and Validation
```bash
# Run tensor-only tests
swipl -s tensor_only_test.pl

# Run full tensor tests  
swipl -s tensor_test.pl

# Syntax checking
swipl -s syntax_check.pl
```

### Docker Operations
```bash
# View container logs
docker-compose logs -f ml-notebook

# Access container shell
docker-compose exec ml-notebook bash

# Stop services
docker-compose down

# Rebuild image
docker-compose build --no-cache
```

### Jupyter Environment
- Access at `http://localhost:8888` (get token from logs)
- Main notebook: `notebooks/Chapter15-Backpropagation-Tensores-Python.ipynb`
- Data files located in `data/` directory

## Key Implementation Details

### Tensor Optimization Focus
- All neural network operations use MXNet tensors instead of Python loops
- Vectorized operations for better performance and GPU compatibility
- Maintains original API compatibility while using tensor backends

### Random Seed Configuration
```python
# Set in notebooks and modules for reproducibility
mx.random.seed(1)
random.seed(1)
```

### Data Format
- Datasets expected as lists of lists: `[[x1, x2, ..., class], ...]`
- Last column is typically the target class
- Supports both binary and multi-class classification

### Performance Benefits
- 5-10x faster than traditional loop-based implementations
- Better memory efficiency
- Preparation for GPU acceleration
- Automatic parallelization of operations

## Common Issues

### MXNet Installation
If MXNet fails to install, use specific version:
```bash
pip install mxnet==1.9.1
```

### Jupyter Graphics
For plot display issues:
```bash
jupyter nbextension enable --py widgetsnbextension
```