import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Кривые обучения (Binary Cross-Entropy)")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, X_train, Y_train, X_test, Y_test):
    plt.figure()
    plt.title("Визуализация разделяющей границы")
    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap="bwr", label="train", marker="o")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap="bwr", label="test", marker="x")

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--")
    plt.legend()
    plt.show()


def plot_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=20)
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--")
    accuracy = np.mean(model.predict(X) == y)
    ax.set_title(f"{title}\nAcc: {accuracy:.2f}")


def plot_roc_and_errors(model, X_test, Y_test, fpr, tpr, auc_val, accuracy, precision, recall):
    y_pred_test = model.predict(X_test)
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    ax[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auc_val:.3f})")
    ax[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("ROC Curve")
    ax[0].legend(loc="lower right")
    ax[0].grid(alpha=0.3)

    ax[1].scatter(
        X_test[(Y_test == 0) & (y_pred_test == 0), 0],
        X_test[(Y_test == 0) & (y_pred_test == 0), 1],
        c="blue",
        alpha=0.3,
        label="Class 0 Correct",
        edgecolors="k",
    )
    ax[1].scatter(
        X_test[(Y_test == 1) & (y_pred_test == 1), 0],
        X_test[(Y_test == 1) & (y_pred_test == 1), 1],
        c="red",
        alpha=0.3,
        label="Class 1 Correct",
        edgecolors="k",
    )
    ax[1].scatter(
        X_test[(Y_test == 0) & (y_pred_test == 1), 0],
        X_test[(Y_test == 0) & (y_pred_test == 1), 1],
        c="cyan",
        marker="X",
        s=100,
        label="False Positive (E0)",
        edgecolors="k",
    )
    ax[1].scatter(
        X_test[(Y_test == 1) & (y_pred_test == 0), 0],
        X_test[(Y_test == 1) & (y_pred_test == 0), 1],
        c="orange",
        marker="X",
        s=100,
        label="False Negative (E1)",
        edgecolors="k",
    )

    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax[1].contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--")
    ax[1].set_title(
        f"Detailed Errors Analysis\nAcc: {accuracy:.2f} | Prec: {precision:.2f} | "
        f"Rec: {recall:.2f} | AUC: {auc_val:.2f}"
    )
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")
    ax[1].legend(loc="best", fontsize="small")
    ax[1].grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
