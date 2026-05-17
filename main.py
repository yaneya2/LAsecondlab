import numpy as np
import matplotlib.pyplot as plt

from dataset import X_train, X_test, Y_train, Y_test
from perceptron import Perceptron


model = Perceptron(seed=42)
train_losses, val_losses = model.fit(
    X_train, Y_train,
    X_test, Y_test,
    epochs=100,
    lr=0.1,
    batch_size=32
)

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Кривые обучения (Binary Cross-Entropy)')

plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

train_acc = np.mean(model.predict(X_train) == Y_train)
test_acc = np.mean(model.predict(X_test) == Y_test)

print(f"\nAccuracy (Train): {train_acc:.4f}")
print(f"Accuracy (Test):  {test_acc:.4f}")

plt.title('Визуализируйте разделяющую границу')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap="bwr", label="train", marker="o")
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap="bwr", label='test', marker="x")

x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Вычисление предсказаний
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.forward(grid_points)
Z = Z.reshape(xx.shape)

# Разделяющая граница
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--')

plt.show()
