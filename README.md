# Attack-on-Learning-Based-Community-Search

## Overview

This repository presents a comprehensive framework for studying adversarial attacks against learning-based community search models in networks. Community search is a fundamental problem in graph analysis, aiming to identify densely connected subgraphs (communities) centered at query nodes. This project systematically investigates the vulnerability of state-of-the-art Graph Neural Network (GNN)-based community detection methods to evasion attacks and proposes robust countermeasures.

## Key Contributions

- **Comprehensive Vulnerability Analysis**: Systematic evaluation of adversarial attacks against multiple learning-based community search architectures
- **Unified Attack Framework**: Implementation of the EVA (Evasion Attack) framework with reinforcement learning-based edge perturbation strategies
- **Multi-Dataset Evaluation**: Support for 8+ real-world social network and citation network datasets
- **Comparative Benchmarking**: Extensive experimental comparison across multiple baseline models and attack scenarios

## Supported Models

### Victim Models (Community Search)

| Model | Type | Key Feature | 
|-------|------|------------|
| **COCLEP** | Contrastive Learning | Hypergraph convolution with contrastive objectives |
| **CSGNN** | Graph Neural Network | Standard community search GNN baseline |
| **ICSGNN** | Improved GNN | Enhanced CSGNN with better aggregation |
| **QDGNN** | Query-Driven GNN | Query-aware network embeddings |
| **CGNP** | Neural Process | Community detection as neural process |
| **CAF** | Augmentation Framework | Community-aware feature augmentation |
| **SUR** | Surrogate Model | Meta-model for black-box attacks |

### Attack Models (Adversarial)

- **EVA**: Evasion Attack Framework with policy-based edge perturbation learning

## Project Structure

```
Attack-on-Learning-Based-Community-Search/
├── args/                          # Argument parsing and configuration
│   ├── args.py                   # Command-line arguments definition
│   └── readme.txt
├── data/                          # Data processing and loading
│   ├── dataset_analysis.py        # Dataset statistics and analysis
│   ├── preprocess_*.py            # Dataset preprocessing scripts
│   └── amazon/                    # Sample Amazon dataset files
├── dataloader/                    # Custom data loaders
│   ├── dataloader_task.py         # Task-based data loading
│   ├── dataloader_task_cgnp.py    # CGNP-specific data loader
│   └── readme.txt
├── loss_criteria/                 # Custom loss functions
│   ├── loss.py                   # Weighted BCE and KL divergence losses
│   └── readme.txt
├── model/                         # Model implementations
│   ├── Vmodel/                   # Victim models (community search)
│   │   ├── COCLEP.py            # Contrastive learning-based model
│   │   ├── CSGNN.py             # Graph neural network variant
│   │   ├── ICSGNN.py            # Improved CSGNN
│   │   ├── QDGNN.py             # Query-driven GNN
│   │   ├── CGNP.py              # Graph neural process
│   │   └── CAF.py               # Community augmentation framework
│   ├── Amodel/                   # Attack models
│   │   └── EVA/                 # Evasion attack framework
│   │       └── Evasion_attack.py # Main attack implementation
│   └── readme.txt
├── sample/                        # Raw data handling
│   ├── RawData.py                # Raw graph data structure
│   └── readme.txt
├── train_model/                   # Training scripts
│   ├── train_coclep.py           # COCLEP training pipeline
│   ├── train_csgnn.py            # CSGNN training
│   ├── train_icsgnn.py           # ICSGNN training
│   ├── train_qdgnn.py            # QDGNN training
│   ├── train_cgnp.py             # CGNP training
│   ├── train_caf.py              # CAF training
│   ├── train_sur.py              # Surrogate model training
│   └── readme.txt
├── utils/                         # Utility functions
│   ├── utils.py                  # Graph utilities, metrics, etc.
│   └── readme.txt
├── main.py                        # Main execution script
└── README.md                      # This file
```

## Datasets

The project supports multiple benchmark datasets:

| Dataset | Type | Nodes | Edges | Communities | Domain |
|---------|------|-------|-------|------------|--------|
| **Amazon** | Co-purchasing | ~335K | ~925K | 5,000 | E-commerce |
| **DBLP** | Citation | ~317K | ~1.05M | 10,000 | Academic |
| **Facebook** | Social | ~63.7K | ~817K | 10,000 | Social |
| **Football** | Sports | 115 | 613 | 12 | Sports |
| **Email** | Communication | 1,005 | 25K | 42 | Organization |
| **LiveJournal** | Social | 4.8M | 68.5M | 5,000 | Social |
| **Cora** | Citation | ~2,708 | ~5,278 | Variable | Academic |
| **CiteSeer** | Citation | ~3,312 | ~4,732 | Variable | Academic |

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- PyTorch Geometric (torch_geometric)
- NumPy
- scikit-learn
- NetworkX

### Setup

```bash
# Clone the repository
git clone https://github.com/.....git
cd Attack-on-Learning-Based-Community-Search

# Install dependencies
pip install torch torch_geometric networkx scikit-learn numpy

# Optional: Install with GPU support
pip install torch-cuda  # Adjust based on your CUDA version
```

## Usage

### Basic Training

To train a community search model on a dataset:

```bash
python main.py \
    --vmodel coclep \
    --data_set amazon \
    --epochs 10 \
    --batch_size 5 \
    --num_layers 3
```

### Configuration

Key arguments (see `args/args.py` for complete list):

- `--vmodel`: Victim model selection (coclep, csgnn, icsgnn, etc.)
- `--smodel`: Surrogate model selection
- `--data_set`: Dataset to use (amazon, dblp, facebook, etc.)
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--num_layers`: Number of GNN layers
- `--learning_rate`: Learning rate for optimization
- `--dropout`: Dropout rate for regularization
- `--task_num`: Number of training tasks
- `--sample_method`: Subgraph sampling method (BFS or RandomWalk)

### Attack Evaluation

To evaluate adversarial attacks against a trained model:

```bash
python main.py \
    --vmodel coclep \
    --attack eva \
    --perturbation_budget 0.1 \
    --attack_epochs 100
```

### Validation and Testing

```bash
python main.py \
    --vmodel coclep \
    --data_set amazon \
    --train False \
    --validation True \
    --test True
```

## Methodology

### Community Search

The victim models perform community search by:
1. **Graph Representation**: Converting networks into node/edge feature representations
2. **Neural Processing**: Using Graph Neural Networks (GCN, GAT, HypergraphConv) to generate embeddings
3. **Community Detection**: Predicting community membership based on learned representations
4. **Ranking**: Scoring and ranking candidate communities for query nodes

### Adversarial Attacks

The attack framework implements:
1. **Edge Perturbations**: Adding or removing edges to manipulate community structure
2. **Feature Perturbations**: Modifying node/edge features to fool the model
3. **Adaptive Attacks**: Iteratively updating perturbations to maximize attack success
4. **Black-box Attacks**: Attacks without knowledge of target model weights

## Performance Metrics

The framework evaluates models using:

- **Accuracy**: Overall prediction correctness
- **Precision/Recall**: Per-class prediction quality
- **F1-Score**: Harmonic mean of precision and recall
- **Attack Success Rate**: Percentage of successful adversarial perturbations


## Experimental Results

Models are evaluated on:
- Prediction accuracy under benign conditions
- Robustness to adversarial attacks
- Computational efficiency and scalability
- Performance across different network densities and community structures

## Key Findings

- GNN-based community search models are vulnerable to targeted adversarial attacks
- Attack effectiveness varies significantly across different community structures
- Ensemble methods provide improved robustness compared to single models


## References

### Keywords

- Community Search in Networks
- Graph Neural Networks  
- Adversarial Attacks on Graphs
- Robustness of GNNs


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch Geometric team for excellent graph neural network implementations
- SNAP database for providing benchmark network datasets
- All researchers who contributed to community detection and adversarial robustness literature


## Changelog

### Version 1.1.3
- Initial release with 7 victim models
- EVA attack framework
- Support for 8 benchmark datasets
- Comprehensive evaluation metrics

---

