import numpy as np
from word2vec.vocab import Vocabulary
from word2vec import config


def load_embeddings(path: str = config.MODEL_PATH) -> np.ndarray:
    return np.load(path)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


def most_similar(word: str, embeddings: np.ndarray, vocab: Vocabulary, top_n: int = 10):
    if word not in vocab.word2idx:
        print(f"'{word}' not in vocabulary")
        return []

    idx = vocab.word2idx[word]
    vec = embeddings[idx]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    scores = normed @ vec / (np.linalg.norm(vec) + 1e-10)

    top_indices = np.argsort(-scores)[:top_n + 1]
    results = [(vocab.idx2word[i], float(scores[i])) for i in top_indices if i != idx][:top_n]
    return results


def analogy(pos1: str, neg1: str, pos2: str, embeddings: np.ndarray, vocab: Vocabulary, top_n: int = 5):
    """Classic word analogy: pos1 - neg1 + pos2 = ?"""
    words = [pos1, neg1, pos2]
    if any(w not in vocab.word2idx for w in words):
        missing = [w for w in words if w not in vocab.word2idx]
        print(f"Words not in vocabulary: {missing}")
        return []

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms

    target = (normed[vocab.word2idx[pos1]]
            - normed[vocab.word2idx[neg1]]
            + normed[vocab.word2idx[pos2]])

    scores = normed @ target
    top_indices = np.argsort(-scores)[:top_n + 3]
    results = [(vocab.idx2word[i], float(scores[i]))
               for i in top_indices
               if vocab.idx2word[i] not in words][:top_n]
    return results


def get_test_words(vocab: Vocabulary) -> list[str]:
    candidates = config.EVAL_TEST_WORDS_DOMAIN if config.USE_TOPIC_KEYWORDS else config.EVAL_TEST_WORDS_GENERAL
    present = [w for w in candidates if w in vocab.word2idx]
    missing = [w for w in candidates if w not in vocab.word2idx]
    if missing:
        print(f"Note: {len(missing)} test word(s) not in vocab: {missing}")
    return present


def get_analogies(vocab: Vocabulary) -> list[tuple[str, str, str]]:
    candidates = config.EVAL_ANALOGIES_DOMAIN if config.USE_TOPIC_KEYWORDS else config.EVAL_ANALOGIES_GENERAL
    valid, skipped = [], []
    for triple in candidates:
        (valid if all(w in vocab.word2idx for w in triple) else skipped).append(triple)
    for triple in skipped:
        print(f"Skipping analogy {triple} — missing: {[w for w in triple if w not in vocab.word2idx]}")
    return valid


if __name__ == "__main__":
    print(f"Loading embeddings from {config.MODEL_PATH}...")
    embeddings = load_embeddings()
    vocab = Vocabulary.load()

    mode = "domain-specific" if config.USE_TOPIC_KEYWORDS else "general"
    print(f"Embeddings shape: {embeddings.shape}  |  mode: {mode}\n")

    # --- Nearest neighbours ---
    test_words = get_test_words(vocab)
    for word in test_words:
        results = most_similar(word, embeddings, vocab)
        if results:
            neighbours = ", ".join([f"{w} ({s:.3f})" for w, s in results[:5]])
            print(f"most_similar('{word}'): {neighbours}")

    print()

    # --- Analogy tests ---
    analogies = get_analogies(vocab)
    for pos1, neg1, pos2 in analogies:
        results = analogy(pos1, neg1, pos2, embeddings, vocab)
        if results:
            answers = ", ".join([f"{w} ({s:.3f})" for w, s in results[:3]])
            print(f"'{pos1}' - '{neg1}' + '{pos2}' = {answers}")
