import numpy as np


class Perceptron:
    def __init__(self, seed=42):
        np.random.seed(seed)

        self.w = np.random.randn(2) * 1e-2
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # чтобы не выдавало бессконечности
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X_train, Y_train, X_val, Y_val, epochs=100, lr=0.01, batch_size=64):
        train_losses = []
        val_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                Y_batch = Y_shuffled[i: i + batch_size]
                m = X_batch.shape[0]

                Y_pred = self.forward(X_batch)

                error = Y_pred - Y_batch
                dw = 1 / m * (X_batch.T @ error)
                db = 1 / m * np.sum(error)

                self.w = self.w - lr * dw
                self.b = self.b - lr * db

                train_pred = self.forward(X_train)
                val_pred = self.forward(X_val)

            train_losses.append(self.compute_loss(Y_train, train_pred))
            val_losses.append(self.compute_loss(Y_val, val_pred))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
        return train_losses, val_losses

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


class PerceptronExtended(Perceptron):
    def __init__(self, seed=42, l2_lambda=0.0):
        super().__init__(seed)
        self.l2_lambda = l2_lambda

    def fit_extended(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        lr=0.1,
        batch_size=32,
        loss_type="bce",
    ):
        if loss_type not in {"bce", "hinge"}:
            raise ValueError("loss_type must be 'bce' or 'hinge'")

        train_losses = []
        w_norms = []
        n_samples = X_train.shape[0]
        y_target = np.where(y_train == 0, -1, 1) if loss_type == "hinge" else y_train

        for _ in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_target[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]
                m = X_batch.shape[0]
                z = X_batch @ self.w + self.b

                if loss_type == "bce":
                    error = self.sigmoid(z) - y_batch
                    dw = (X_batch.T @ error) / m + self.l2_lambda * self.w
                    db = np.sum(error) / m
                else:
                    mask = (y_batch * z < 1).astype(float)
                    dw = X_batch.T @ (-mask * y_batch) / m + self.l2_lambda * self.w
                    db = np.sum(-mask * y_batch) / m

                self.w -= lr * dw
                self.b -= lr * db

            z_all = X_train @ self.w + self.b
            regularization = 0.5 * self.l2_lambda * np.sum(self.w**2)
            if loss_type == "bce":
                loss = self.compute_loss(y_train, self.sigmoid(z_all)) + regularization
            else:
                loss = np.mean(np.maximum(0, 1 - y_target * z_all)) + regularization

            train_losses.append(loss)
            w_norms.append(np.linalg.norm(self.w))

        return train_losses, w_norms


class PerceptronWithMomentum(Perceptron):
    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        lr=0.1,
        batch_size=32,
        momentum=0.9,
    ):
        train_losses = []
        val_losses = []
        n_samples = X_train.shape[0]
        v_w = np.zeros_like(self.w)
        v_b = 0.0

        for _ in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]
                m = X_batch.shape[0]
                error = self.forward(X_batch) - y_batch
                dw = (X_batch.T @ error) / m
                db = np.sum(error) / m

                v_w = momentum * v_w - lr * dw
                v_b = momentum * v_b - lr * db
                self.w += v_w
                self.b += v_b

            train_loss = self.compute_loss(y_train, self.forward(X_train))
            val_loss = self.compute_loss(y_val, self.forward(X_val))
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if np.isnan(train_loss):
                break

        return train_losses
