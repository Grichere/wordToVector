import pickle, os
from word2vec import config

def load_corpus(path: str = config.RAW_PATH) -> list[list[str]]:
    """Reads cleaned text file, returns list of tokenised sentences."""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences

def load_or_cache(processed_path: str = config.PROCESSED_PATH):
    """Cache the tokenised corpus so you don't re-read the file every run."""
    if os.path.exists(processed_path):
        print("Loading cached corpus...")
        with open(processed_path, "rb") as f:
            return pickle.load(f)

    print("Tokenising corpus from scratch...")
    sentences = load_corpus()
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    with open(processed_path, "wb") as f:
        pickle.dump(sentences, f)
    print(f"Cached {len(sentences)} sentences to {processed_path}")
    return sentences

if __name__ == "__main__":
    sentences = load_or_cache()
    total_tokens = sum(len(s) for s in sentences)
    print(f"Sentences: {len(sentences):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Sample: {sentences[0]}")
    print(f"Avg sentence length: {total_tokens / len(sentences):.1f} tokens")

