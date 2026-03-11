EMBEDDING_DIM = 100
WINDOW_SIZE = 5
MIN_COUNT = 5
MAX_VOCAB_SIZE = 100_000
NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.025
EPOCHS = 5
SUBSAMPLE_THRESHOLD = 1e-5

#Paths
RAW_PATH = "data/raw/wiki_simple.txt"
PROCESSED_PATH = "data/processed/corpus.pkl"
MODEL_PATH = "data/model/embeddings.npy"
VOCAB_PATH = "data/model/vocab.pkl"

#Data download specifications
URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
TOPIC_KEYWORDS = ["animal", "species", "mammal", "biology", "bird", "fish"]

#training params
LOG_EVERY = 10_000
SEED = 42