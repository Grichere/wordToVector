import numpy as np
import time
from word2vec import config
from word2vec.vocab import Vocabulary
from word2vec.dataset import generate_pairs
from word2vec.model import SkipGramModel


def train(model: SkipGramModel, sentences: list[list[str]], vocab: Vocabulary):
    np.random.seed(config.SEED)

    total_loss = 0.0
    pair_count = 0
    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_loss = 0.0
        epoch_pairs = 0

        for centre_idx, context_idx in generate_pairs(sentences, vocab):
            # Sample K negative indices from noise distribution
            # Replace=True allows same word to be sampled twice (standard)
            neg_indices = np.random.choice(
                len(vocab),
                size=config.NEGATIVE_SAMPLES,
                replace=True,
                p=vocab.noise_dist
            )

            # Avoid using the centre or context word as a negative
            neg_indices = neg_indices[
                (neg_indices != centre_idx) & (neg_indices != context_idx)
            ]
            # If filtering removed too many, just pad back with random samples
            while len(neg_indices) < config.NEGATIVE_SAMPLES:
                neg_indices = np.append(neg_indices, np.random.randint(len(vocab)))

            loss = model.forward_and_grad(centre_idx, context_idx, neg_indices)
            epoch_loss += loss
            epoch_pairs += 1

            if epoch_pairs % config.LOG_EVERY == 0:
                avg_loss = epoch_loss / epoch_pairs
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch} | Pairs: {epoch_pairs:,} | Avg loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

        total_loss += epoch_loss
        pair_count += epoch_pairs
        print(f"Epoch {epoch}/{config.EPOCHS} complete — loss: {epoch_loss/epoch_pairs:.4f} | pairs: {epoch_pairs:,}")

    print(f"\nTraining complete. Total pairs: {pair_count:,} | Overall avg loss: {total_loss/pair_count:.4f}")
