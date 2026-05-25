import numpy as np

from dataset import X_train, X_test, Y_train, Y_test
from perceptron import Perceptron
from visualization import plot_decision_boundary, plot_learning_curves


def main():
    model = Perceptron(seed=42)
    train_losses, val_losses = model.fit(
        X_train,
        Y_train,
        X_test,
        Y_test,
        epochs=100,
        lr=0.1,
        batch_size=32,
    )

    plot_learning_curves(train_losses, val_losses)

    train_acc = np.mean(model.predict(X_train) == Y_train)
    test_acc = np.mean(model.predict(X_test) == Y_test)
    print(f"\nAccuracy (Train): {train_acc:.4f}")
    print(f"Accuracy (Test):  {test_acc:.4f}")

    plot_decision_boundary(model, X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()
