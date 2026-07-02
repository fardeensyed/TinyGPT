# 🎯 TinyGPT

**A minimal, educational implementation of a GPT-like transformer language model from scratch using PyTorch.**

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Overview

TinyGPT is a lightweight, from-scratch implementation of a transformer-based language model inspired by GPT architecture. It demonstrates the core concepts of modern language models including:

- **Self-Attention Mechanisms** - Multi-head attention for parallel representation learning
- **Positional Embeddings** - Token and position encoding for sequence understanding
- **Transformer Blocks** - Combination of attention and feed-forward layers with residual connections
- **Text Generation** - Autoregressive token prediction for generating new sequences

This project is ideal for:
- Learning how transformer language models work internally
- Understanding attention mechanisms in practice
- Experimenting with minimal language model implementations
- Educational purposes and research prototyping

## Architecture

```
TinyGPT
├── Token Embedding Layer
├── Positional Embedding Layer
├── Transformer Blocks (N layers)
│   ├── Multi-Head Self-Attention
│   ├── Layer Normalization
│   ├── Feed-Forward Network
│   └── Residual Connections
├── Final Layer Normalization
└── Output Head (Vocabulary Projection)
```

## Project Structure

```
TinyGPT/
├── demo.py                    # Main training and inference script
├── transformer_blocks.py      # Core transformer components
└── README.md                  # This file
```

## Components

### `transformer_blocks.py`
Contains the building blocks of the transformer architecture:

- **`SelfAttentionHead`** - Single attention head with query, key, and value projections
- **`MultiHeadAttention`** - Parallel attention heads with concatenation and projection
- **`FeedForward`** - Position-wise feed-forward network (FFN)
- **`Block`** - Complete transformer block combining attention and FFN with residual connections

### `demo.py`
Main script featuring:

- Text corpus preparation and vocabulary building
- Batch data generation
- Model training loop
- Text generation pipeline

## Getting Started

### Prerequisites

```bash
pip install torch numpy
```

### Usage

Run the demo to train and generate text:

```bash
python demo.py
```

#### Example Output
```
Torch version: 2.0.1
CUDA available: True
GPU name: NVIDIA GeForce RTX 4090

Training model...
[Training progress with loss values]

Generated text:
hello friends how are you the tea is very <END> diwali brings lights...
```

## Model Configuration

Default hyperparameters (configurable in `demo.py`):

```python
block_size = 6              # Context window size
embedding_dim = 32          # Embedding dimension
n_heads = 2                 # Number of attention heads
n_layers = 2                # Number of transformer blocks
lr = 1e-3                   # Learning rate
epochs = 1500               # Training epochs
batch_size = 16             # Batch size
```

## How It Works

### 1. **Tokenization & Vocabulary**
- Input text is split into words
- Each word is mapped to a unique index
- Special `<END>` token marks sentence boundaries

### 2. **Embedding**
- Token indices are converted to dense vectors (token embeddings)
- Position indices are converted to position embeddings
- Token and position embeddings are summed for contextualized representations

### 3. **Transformer Processing**
- Input passes through N transformer blocks
- Each block applies:
  - Multi-head self-attention (learns word relationships)
  - Residual connection (adds input to attention output)
  - Layer normalization (stabilizes training)
  - Feed-forward network (non-linear transformations)
  - Residual connection (adds input to FFN output)

### 4. **Output & Loss**
- Final layer normalization
- Linear projection to vocabulary size
- Cross-entropy loss between predictions and targets

### 5. **Generation**
- Starting from initial tokens, model predicts next token
- Prediction is sampled probabilistically
- Process repeats to generate sequences of desired length

## Key Features

✅ **Pure PyTorch Implementation** - No external libraries, full transparency  
✅ **GPU Support** - Automatic CUDA detection and utilization  
✅ **Causal Attention** - Prevents attending to future tokens (autoregressive)  
✅ **Residual Connections** - Improves training stability and gradient flow  
✅ **Modular Design** - Easy to extend and experiment with  
✅ **Educational Focus** - Clean, well-commented code  

## Training Details

- **Optimizer**: Adam (implicit in standard approach, can be added)
- **Loss Function**: Cross-entropy loss
- **Batch Processing**: Random sampling for improved generalization
- **Data Format**: Sequences of length `block_size + 1` (input + target)

## Extending TinyGPT

Ideas to experiment with:

1. **Increase Model Size** - More layers, larger embeddings, more heads
2. **Larger Corpus** - Real text datasets for better language understanding
3. **Add Optimizers** - Implement Adam or other advanced optimizers
4. **Dropout Regularization** - Prevent overfitting
5. **Vocabulary Expansion** - Character-level or BPE tokenization
6. **Evaluation Metrics** - Perplexity, BLEU score, etc.
7. **Save/Load Checkpoints** - Model persistence

## Performance Notes

- Model trains efficiently on CPU or GPU
- Current configuration is suitable for demonstrations and learning
- For larger models, consider:
  - Gradient accumulation
  - Mixed precision training
  - Flash attention optimizations

## Educational Resources

This implementation is based on transformer architecture principles from:
- "Attention is All You Need" (Vaswani et al., 2017)
- GPT model series (OpenAI)
- Modern deep learning best practices

## Limitations

- Small vocabulary and simple corpus (single-word tokenization)
- Minimal training data (10 example sentences)
- No advanced features (dropout, weight decay, learning rate scheduling)
- For production use, consider using larger models and frameworks like Hugging Face

## License

MIT License - Feel free to use, modify, and distribute.

## Contributing

Contributions and suggestions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share experiments and results

## Author

Created as an educational exploration of transformer language models.

---

**Happy Learning! 🚀**

Start experimenting with TinyGPT and build your understanding of modern language models from the ground up!
