import numpy as np
from word2vec import config
from word2vec.corpus import load_or_cache
from word2vec.vocab import Vocabulary
from word2vec.model import SkipGramModel
from word2vec.trainer import train
import os


def main():
    np.random.seed(config.SEED)
    print(f"=== Word2Vec Training (version: {config.VERSION}) ===\n")

    # Step 1: Load corpus
    print("Step 1: Loading corpus...")
    sentences = load_or_cache()
    total_tokens = sum(len(s) for s in sentences)
    print(f"  {len(sentences):,} sentences | {total_tokens:,} tokens\n")

    # Step 2: Build vocabulary
    print("Step 2: Building vocabulary...")
    if os.path.exists(config.VOCAB_PATH):
        print("  Loading cached vocab...")
        vocab = Vocabulary.load()
    else:
        vocab = Vocabulary(sentences)
        vocab.save()
    print(f"  Vocab size: {len(vocab):,}\n")

    # Step 3: Initialise model
    print("Step 3: Initialising model...")
    model = SkipGramModel(vocab_size=len(vocab))
    print(f"  W_in:  {model.W_in.shape}")
    print(f"  W_out: {model.W_out.shape}")
    mem_mb = (model.W_in.nbytes + model.W_out.nbytes) / 1e6
    print(f"  Memory: {mem_mb:.1f} MB\n")

    # Step 4: Train
    print("Step 4: Training...\n")
    train(model, sentences, vocab)

    # Step 5: Save embeddings
    print("\nStep 5: Saving embeddings...")
    model.save()
    print("Done.")


if __name__ == "__main__":
    main()
