import numpy as np
from word2vec import config


class SkipGramModel:
    def __init__(self, vocab_size: int):
        self.V = vocab_size
        self.d = config.EMBEDDING_DIM

        # Initialise small random weights — standard practice
        # W_in:  each row is the input  vector for a word (what it looks like as centre)
        # W_out: each row is the output vector for a word (what it looks like as context)
        self.W_in  = np.random.uniform(-0.5 / self.d, 0.5 / self.d, (self.V, self.d)).astype(np.float32)
        self.W_out = np.zeros((self.V, self.d), dtype=np.float32)

    # ------------------------------------------------------------------
    # Core maths
    # ------------------------------------------------------------------

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in exp for very large/small values
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def forward_and_grad(
        self,
        centre_idx: int,
        context_idx: int,
        neg_indices: np.ndarray
    ) -> float:
        """
        Computes loss and applies SGD updates for one (centre, context) pair
        with K negative samples.

        Returns the scalar loss for logging.
        """
        # --- Forward pass ---
        h = self.W_in[centre_idx]                          # shape (d,)  — hidden vector

        # Positive sample: label = 1
        score_pos = self.sigmoid(np.dot(self.W_out[context_idx], h))
        loss = -np.log(score_pos + 1e-10)

        # Negative samples: label = 0
        scores_neg = self.sigmoid(np.dot(self.W_out[neg_indices], h))  # shape (K,)
        loss += -np.sum(np.log(1.0 - scores_neg + 1e-10))

        # --- Gradients ---
        # Error signals: e = sigma(score) - label
        e_pos = score_pos - 1.0                            # scalar,   label=1
        e_neg = scores_neg                                 # shape (K,), label=0

        # Gradient w.r.t. h (accumulate from positive and all negatives)
        # dL/dh = e_pos * W_out[context] + sum(e_neg_k * W_out[neg_k])
        grad_h  = e_pos * self.W_out[context_idx]
        grad_h += np.dot(e_neg, self.W_out[neg_indices])   # shape (d,)

        # --- SGD updates ---
        # Output vectors
        self.W_out[context_idx] -= config.LEARNING_RATE * e_pos * h
        self.W_out[neg_indices] -= config.LEARNING_RATE * np.outer(e_neg, h)

        # Input vector for centre word
        self.W_in[centre_idx]   -= config.LEARNING_RATE * grad_h

        return float(loss)

    # ------------------------------------------------------------------
    # Final embeddings
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> np.ndarray:
        """Standard practice: average input and output embeddings."""
        return (self.W_in + self.W_out) / 2.0

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    def save(self, path: str = config.MODEL_PATH):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.embeddings)
        print(f"Embeddings saved to {path}")

    # ------------------------------------------------------------------
    # Gradient check (finite differences) — run this before full training
    # ------------------------------------------------------------------

    def gradient_check(self, centre_idx: int, context_idx: int, neg_indices: np.ndarray):
        """
        Numerically verify analytic gradients match finite difference estimates.
        Should only be called on a small model for debugging.
        """
        eps = 1e-4

        # Save original weights
        W_in_orig  = self.W_in.copy()
        W_out_orig = self.W_out.copy()

        def loss_fn():
            h = self.W_in[centre_idx]
            s_pos = self.sigmoid(np.dot(self.W_out[context_idx], h))
            s_neg = self.sigmoid(np.dot(self.W_out[neg_indices], h))
            return -np.log(s_pos + 1e-10) - np.sum(np.log(1 - s_neg + 1e-10))

        print("Gradient check on W_in[centre_idx]:")
        analytic_grad = np.zeros(self.d)

        # Compute analytic gradient for W_in[centre]
        h = self.W_in[centre_idx]
        e_pos = self.sigmoid(np.dot(self.W_out[context_idx], h)) - 1.0
        e_neg = self.sigmoid(np.dot(self.W_out[neg_indices], h))
        analytic_grad = e_pos * self.W_out[context_idx] + np.dot(e_neg, self.W_out[neg_indices])

        # Compute numerical gradient
        numerical_grad = np.zeros(self.d)
        for i in range(self.d):
            self.W_in[centre_idx][i] += eps
            loss_plus = loss_fn()
            self.W_in[centre_idx][i] -= 2 * eps
            loss_minus = loss_fn()
            self.W_in[centre_idx][i] += eps   # restore
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)

        diff = np.max(np.abs(analytic_grad - numerical_grad))
        print(f"  Max absolute difference: {diff:.2e}  ({'OK' if diff < 1e-4 else 'FAIL'})")

        # Restore weights
        self.W_in  = W_in_orig
        self.W_out = W_out_orig


if __name__ == "__main__":
    # Quick test on a tiny model
    np.random.seed(config.SEED)
    model = SkipGramModel(vocab_size=100)
    centre = 0
    context = 5
    negs = np.array([10, 20, 30, 40, 50])

    model.gradient_check(centre, context, negs)

    loss = model.forward_and_grad(centre, context, negs)
    print(f"Loss on dummy input: {loss:.4f}")
    print(f"Embeddings shape: {model.embeddings.shape}")
