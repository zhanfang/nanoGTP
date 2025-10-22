# Shakespeare GPT

A character-level GPT (Generative Pre-trained Transformer) model trained on Shakespeare's works. This project implements a small but capable GPT model that can generate Shakespeare-style text.

## Project Structure

```
├── configs/         # Training configurations
├── data/           
│   └── shakespeare/  # Shakespeare dataset
├── model.py        # GPT model implementation
├── prepare.py      # Data preprocessing utilities
├── sample.py       # Text generation script
├── train.py        # Training script
└── requirements.txt # Project dependencies
```

## Setup

1. Create a virtual environment and activate it:
```bash
python3 -m venv .env
source .env/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. First, download the Shakespeare dataset:
```bash
mkdir -p data/shakespeare
curl -o data/shakespeare/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

2. Prepare the training data:
```bash
python prepare.py
```

3. Train the model:
```bash
python train.py
```

The model will be saved as `model.pth` after training.

### Text Generation

To generate Shakespeare-style text using the trained model:

```bash
python sample.py
```

You can then:
1. Enter a starting text prompt (e.g., "ROMEO:")
2. Specify how many tokens to generate
3. Set the temperature (0.0-1.0):
   - Lower values (~0.3) for more focused text
   - Higher values (~1.0) for more creative text

## Model Architecture

The model is a small GPT implementation with:
- Multi-head self-attention
- Layer normalization
- Feed-forward networks
- Positional embeddings

Key hyperparameters:
- Embedding dimension: 384
- Number of layers: 6
- Number of attention heads: 6
- Context size: 256 tokens
- Vocabulary size: 65 characters

## Example Output

```
Enter some starting text: ROMEO:
How many tokens to generate? 100
Enter temperature (0.0-1.0): 0.7

Generated text:
ROMEO:
Alas, for my son, it shall be none.

GLOUCESTER:
What doth speak a shortly as we ever here.

LADY ANNE:
That shall you know hereafter.
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- tqdm

## Notes

- The model is trained on character-level tokens, making it capable of generating any combination of characters present in Shakespeare's works
- The temperature parameter controls the randomness of the generation:
  - 0.0: Always pick the most likely next character
  - 1.0: Sample from the full probability distribution
- The model preserves Shakespeare's formatting, including character names and dialogue structure

## License

MIT