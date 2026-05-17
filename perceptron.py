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
