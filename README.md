# WordToVector

A from-scratch implementation of the Word2Vec skip-gram model with negative sampling, built in pure NumPy. Trains word embeddings on Wikipedia articles and evaluates them using nearest-neighbour and analogy tasks.

## How It Works

- Fetches Wikipedia articles via the Wikipedia API
- Builds a vocabulary with frequency-based subsampling
- Trains a skip-gram model using noise contrastive estimation (negative sampling)
- Evaluates learned embeddings with cosine similarity and word analogies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```python train.py```

## Evaluate
```python evaluate.py```

## Configuration
To be made in the config.py file
