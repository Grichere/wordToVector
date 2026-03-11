import numpy as np
import pickle, os
from collections import Counter
from word2vec import config

class Vocabulary:
    def __init__(self, sentences: list[list[str]]):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_freqs: dict[str, int] = {}
        self.noise_dist: np.ndarray = None

        self._build(sentences)

    def _build(self, sentences: list[list[str]]):
        # Count all token frequencies
        counter = Counter(w for sent in sentences for w in sent)

        # Filter by min_count, sort by frequency descending
        filtered = sorted(
            [(w, f) for w, f in counter.items() if f >= config.MIN_COUNT],
            key=lambda x: -x[1]
        )

        # Trim to max vocab size
        filtered = filtered[:config.MAX_VOCAB_SIZE]

        # Build mappings
        for idx, (word, freq) in enumerate(filtered):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_freqs[word] = freq

        print(f"Vocabulary size: {len(self.word2idx):,} words")
        print(f"Coverage: {self._coverage(counter):.1%} of all tokens")

        # Pre-compute noise distribution P(w)^(3/4)
        freqs = np.array([self.word_freqs[self.idx2word[i]] for i in range(len(self.word2idx))], dtype=np.float32)
        freqs = freqs ** 0.75
        self.noise_dist = freqs / freqs.sum()   # normalise to probabilities

    def _coverage(self, raw_counter: Counter) -> float:
        """Fraction of total tokens covered by vocab."""
        total = sum(raw_counter.values())
        covered = sum(f for w, f in raw_counter.items() if w in self.word2idx)
        return covered / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, path: str = config.VOCAB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Vocab saved to {path}")

    @staticmethod
    def load(path: str = config.VOCAB_PATH) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from word2vec.corpus import load_or_cache
    sentences = load_or_cache()
    vocab = Vocabulary(sentences)
    vocab.save()

    # Sanity checks
    print(f"\nSample words: {list(vocab.word2idx.items())[:10]}")
    print(f"'animal' idx: {vocab.word2idx.get('animal', 'NOT FOUND')}")
    print(f"Noise dist sums to: {vocab.noise_dist.sum():.6f}")   # should be ~1.0
