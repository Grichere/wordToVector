import numpy as np
from word2vec.vocab import Vocabulary
from word2vec import config


def subsample_sentence(sentence: list[str], vocab: Vocabulary, total_tokens: int) -> list[int]:
    """
    Convert sentence to indices, randomly dropping frequent words.
    Returns list of kept word indices.
    """
    kept = []
    for word in sentence:
        if word not in vocab.word2idx:
            continue                        # skip words below min_count
        idx = vocab.word2idx[word]
        freq = vocab.word_freqs[word] / total_tokens
        # keep probability — rare words kept almost always, frequent words dropped often
        p_keep = min(1.0, (np.sqrt(freq / config.SUBSAMPLE_THRESHOLD) + 1) * (config.SUBSAMPLE_THRESHOLD / freq))
        if np.random.rand() < p_keep:
            kept.append(idx)
    return kept


def generate_pairs(sentences: list[list[str]], vocab: Vocabulary):
    """
    Yields (centre_idx, context_idx) pairs for all sentences.
    Uses a random window size 1..WINDOW_SIZE per centre word (standard word2vec trick).
    """
    total_tokens = sum(vocab.word_freqs.values())

    for sentence in sentences:
        indices = subsample_sentence(sentence, vocab, total_tokens)
        if len(indices) < 2:
            continue

        for i, centre in enumerate(indices):
            window = np.random.randint(1, config.WINDOW_SIZE + 1)
            left  = max(0, i - window)
            right = min(len(indices), i + window + 1)

            for j in range(left, right):
                if j == i:
                    continue                # skip the centre word itself
                yield centre, indices[j]


if __name__ == "__main__":
    from word2vec.corpus import load_or_cache
    from word2vec.vocab import Vocabulary

    sentences = load_or_cache()
    vocab = Vocabulary.load()

    # Count pairs from first 100 sentences only
    sample = sentences[:100]
    pairs = list(generate_pairs(sample, vocab))
    print(f"Pairs from 100 sentences: {len(pairs):,}")
    print(f"Sample pair (centre, context): {pairs[0]}")
    print(f"  centre word: {vocab.idx2word[pairs[0][0]]}")
    print(f"  context word: {vocab.idx2word[pairs[0][1]]}")
