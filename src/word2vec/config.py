EMBEDDING_DIM = 100
WINDOW_SIZE = 5
MIN_COUNT = 5
MAX_VOCAB_SIZE = 100_000
NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.025
EPOCHS = 20
SUBSAMPLE_THRESHOLD = 1e-5

#Paths
VERSION = "2"
RAW_PATH = f"data/raw/wiki_simple_{VERSION}.txt"
PROCESSED_PATH = f"data/processed/corpus_{VERSION}.pkl"
MODEL_PATH = f"data/model/embeddings_{VERSION}.npy"
VOCAB_PATH = f"data/model/vocab_{VERSION}.pkl"

#Data download specifications
URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
TOPIC_KEYWORDS = ["animal", "species", "mammal", "biology", "bird", "fish"]
USE_TOPIC_KEYWORDS = True
MAX_ARTICLES = 5000

#training params
LOG_EVERY = 10_000
SEED = 42

#eval
EVAL_TEST_WORDS_DOMAIN = ["fish", "mammal", "plant", "water", "species"]
EVAL_TEST_WORDS_GENERAL = ["king", "man", "woman", "city", "science"]

# A - B + C
EVAL_ANALOGIES_DOMAIN = [
    ("fish",   "water",  "bird"),
    ("mammal", "animal", "plant"),
]
EVAL_ANALOGIES_GENERAL = [
    ("king",   "man",    "woman"),
    ("paris",  "france", "germany"),
    ("walked", "walk",   "run"),
]