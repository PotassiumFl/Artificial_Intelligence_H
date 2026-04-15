import numpy as np

import config


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -500, 500))
    return e / np.sum(e, axis=1, keepdims=True)


class Net(object):
    def __init__(self, input_size, hidden_sizes, output_size, task, seed):
        self.task = task
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self._rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        self._init_weights()

    def _init_weights(self):
        dims = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = self._rng.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.W.append(w)
            self.b.append(b)

    def forward(self, X):
        self._cache = {"a": [X]}
        a = X
        for i in range(len(self.hidden_sizes)):
            z = a @ self.W[i] + self.b[i]
            a = sigmoid(z)
            self._cache.setdefault("z", []).append(z)
            self._cache["a"].append(a)
        z_out = a @ self.W[-1] + self.b[-1]
        self._cache["z_out"] = z_out
        if self.task == "classification":
            out = softmax(z_out)
        else:
            out = z_out
        self._cache["out"] = out
        return out

    def _loss_and_grad_output(self, y_true, batch_size):
        z_out = self._cache["z_out"]
        if self.task == "regression":
            pred = z_out
            diff = pred - y_true
            loss = 0.5 * np.mean(np.sum(diff**2, axis=1))
            grad_z = diff / batch_size
            return loss, grad_z
        prob = softmax(z_out)
        eps = 1e-12
        loss = -np.mean(np.sum(y_true * np.log(prob + eps), axis=1))
        grad_z = (prob - y_true) / batch_size
        return loss, grad_z

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        loss, grad_z = self._loss_and_grad_output(y_true, batch_size)

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        a_list = self._cache["a"]
        z_hidden = self._cache["z"]

        L = len(self.W) - 1
        a_prev = a_list[-1]
        grads_W[L] = a_prev.T @ grad_z
        grads_b[L] = np.sum(grad_z, axis=0, keepdims=True)
        grad_a = grad_z @ self.W[L].T

        for i in range(L - 1, -1, -1):
            z_i = z_hidden[i]
            grad_z = grad_a * sigmoid_prime(z_i)
            a_prev = a_list[i]
            grads_W[i] = a_prev.T @ grad_z
            grads_b[i] = np.sum(grad_z, axis=0, keepdims=True)
            if i > 0:
                grad_a = grad_z @ self.W[i].T

        return loss, grads_W, grads_b

    def step(self, grads_W, grads_b, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * grads_W[i]
            self.b[i] -= lr * grads_b[i]

    def compute_loss(self, X, y):
        self.forward(X)
        batch_size = y.shape[0]
        loss, _ = self._loss_and_grad_output(y, batch_size)
        return loss

    def fit(
        self,
        X,
        y,
        epochs=config.EPOCHS,
        lr=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        verbose=True,
        X_val=None,
        y_val=None,
        early_stop_patience=None,
        early_stop_min_delta=None,
    ):
        if early_stop_patience is None:
            early_stop_patience = config.EARLY_STOP_PATIENCE
        if early_stop_min_delta is None:
            early_stop_min_delta = config.EARLY_STOP_MIN_DELTA

        def _val_loss_improved(val_loss, best_val_loss, rel_delta):
            if best_val_loss == float("inf"):
                return True
            if rel_delta > 0:
                if best_val_loss <= 0:
                    return val_loss < best_val_loss
                return val_loss < best_val_loss * (1.0 - rel_delta)
            return val_loss < best_val_loss

        use_early_stop = (
            X_val is not None and y_val is not None and early_stop_patience > 0
        )
        if use_early_stop:
            best_val_loss = float("inf")
            best_W = None
            best_b = None
            patience_ctr = 0

        n = X.shape[0]
        for ep in range(epochs):
            idx = self._rng.permutation(n)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                xb = X[batch_idx]
                yb = y[batch_idx]
                self.forward(xb)
                loss, gW, gb = self.backward(yb)
                self.step(gW, gb, lr)
                epoch_loss += loss
                steps += 1

            val_loss = self.compute_loss(X_val, y_val)
            if use_early_stop:
                if _val_loss_improved(val_loss, best_val_loss, early_stop_min_delta):
                    best_val_loss = val_loss
                    best_W = [w.copy() for w in self.W]
                    best_b = [b.copy() for b in self.b]
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= early_stop_patience:
                        if verbose:
                            print(
                                f"早停于 epoch {ep + 1}/{epochs}  "
                                f"验证集 loss={best_val_loss:.6f}"
                            )
                        break

            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                line = f"epoch {ep + 1}/{epochs}  loss={epoch_loss / max(steps, 1):.6f}  val_loss={val_loss:.6f}"
                print(line)

        if use_early_stop and best_W is not None:
            self.W = best_W
            self.b = best_b

    def predict(self, X):
        out = self.forward(X)
        if self.task == "classification":
            return np.argmax(out, axis=1)
        return out

    def score(self, X, y):
        pred = self.predict(X)
        if self.task == "classification":
            y_cls = np.argmax(y, axis=1)
            return np.mean(pred == y_cls)
        mse = np.mean((pred - y) ** 2)
        return mse

    def get_state(self):
        return {
            "task": self.task,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_sizes": list(self.hidden_sizes),
            "W": [w.copy() for w in self.W],
            "b": [b.copy() for b in self.b],
        }

    @classmethod
    def from_state(cls, state, seed=0):
        net = cls(
            state["input_size"],
            state["hidden_sizes"],
            state["output_size"],
            state["task"],
            seed,
        )
        net.W = [w.copy() for w in state["W"]]
        net.b = [b.copy() for b in state["b"]]
        return net
