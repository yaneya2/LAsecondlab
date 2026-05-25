import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset import (
    X_test,
    X_train,
    Y_test,
    Y_train,
    generate_circle,
    generate_linear,
    generate_xor,
)
from metrics import compute_f1, compute_precision, compute_recall, compute_roc_auc
from perceptron import Perceptron, PerceptronExtended, PerceptronWithMomentum
from visualization import plot_boundary, plot_roc_and_errors

try:
    from IPython.display import display
except ImportError:
    display = print


def run_parameter_experiments():
    results = []

    learning_rates = [0.001, 0.01, 0.5, 1.0, 2.0, 5.0]
    plt.figure(figsize=(10, 5))
    for cur_lr in learning_rates:
        print(f"current learning rate = {cur_lr}")
        model = Perceptron(seed=42)
        train_losses, val_losses = model.fit(
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=100,
            lr=cur_lr,
            batch_size=32,
        )
        results.append(
            {
                "Exp": "learning rate",
                "param": cur_lr,
                "accuracy": np.mean(model.predict(X_test) == Y_test),
                "final_Train_Loss": train_losses[-1],
                "final_Val_Loss": val_losses[-1],
            }
        )
        print(50 * "-")
        plt.plot(val_losses, label=f"lr={cur_lr}")

    plt.title("Experiment: learning_rate")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    batch_sizes = [1, 16, 64, 256]
    plt.figure(figsize=(10, 5))
    for cur_bs in batch_sizes:
        print(f"current batch size = {cur_bs}")
        model = Perceptron(seed=42)
        train_losses, val_losses = model.fit(
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=100,
            lr=0.1,
            batch_size=cur_bs,
        )
        results.append(
            {
                "Exp": "batch",
                "param": cur_bs,
                "accuracy": np.mean(model.predict(X_test) == Y_test),
                "final_Train_Loss": train_losses[-1],
                "final_Val_Loss": val_losses[-1],
            }
        )
        print(50 * "-")
        plt.plot(val_losses, label=f"batch size={cur_bs}")

    plt.title("Experiment: batch_size")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    init_scales = [0, 1, 10]
    init_names = {0: "zero", 1: "small", 10: "large"}
    plt.figure(figsize=(10, 5))
    for cur_scale in init_scales:
        print(f"current type = {cur_scale}")
        model = Perceptron(seed=42)
        model.w = np.random.randn(2) * cur_scale
        train_losses, val_losses = model.fit(
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=100,
            lr=0.1,
            batch_size=32,
        )
        results.append(
            {
                "Exp": "init_type",
                "param": init_names[cur_scale],
                "accuracy": np.mean(model.predict(X_test) == Y_test),
                "final_Train_Loss": train_losses[-1],
                "final_Val_Loss": val_losses[-1],
            }
        )
        print(50 * "-")
        plt.plot(val_losses, label=f"type={init_names[cur_scale]}")

    plt.title("Experiment: init_type")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    results_df = pd.DataFrame(results)
    display(results_df)
    return results_df


def run_synthetic_data_experiment(noise=0.0):
    datasets = {
        "Linear": generate_linear(noise=noise),
        "XOR": generate_xor(noise=noise),
        "Circle": generate_circle(noise=noise),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, (X, y)) in enumerate(datasets.items()):
        indices = np.random.permutation(len(X))
        split = int(0.7 * len(X))
        train_idx, test_idx = indices[:split], indices[split:]
        model = Perceptron(seed=42)
        model.fit(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
            epochs=100,
            lr=0.1,
            batch_size=16,
        )
        plot_boundary(axes[i], model, X, y, name)

    plt.tight_layout()
    plt.show()


def run_loss_and_regularization_experiment():
    model_bce = PerceptronExtended(seed=42)
    model_hinge = PerceptronExtended(seed=42)
    bce_losses, _ = model_bce.fit_extended(
        X_train,
        Y_train,
        X_test,
        Y_test,
        loss_type="bce",
    )
    hinge_losses, _ = model_hinge.fit_extended(
        X_train,
        Y_train,
        X_test,
        Y_test,
        loss_type="hinge",
    )

    plt.figure(figsize=(10, 4))
    plt.plot(bce_losses, label="BCE Loss")
    plt.plot(hinge_losses, label="Hinge Loss")
    plt.title("Comparison: BCE vs Hinge Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    lambdas = [0.0, 0.1, 1.0]
    results_l2 = []
    plt.figure(figsize=(10, 4))
    for lb in lambdas:
        model_l2 = PerceptronExtended(seed=42, l2_lambda=lb)
        _, norms = model_l2.fit_extended(
            X_train,
            Y_train,
            X_test,
            Y_test,
            lr=0.01,
        )
        accuracy = np.mean(model_l2.predict(X_test) == Y_test)
        results_l2.append(
            {
                "Lambda": lb,
                "Accuracy": accuracy,
                "Final_W_Norm": norms[-1],
            }
        )
        plt.plot(norms, label=f"Lambda={lb}")

    plt.title("Weight Norm ||w|| vs Epochs (L2 Regularization)")
    plt.ylabel("||w||")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    results_l2_df = pd.DataFrame(results_l2)
    display(results_l2_df)
    return results_l2_df


def run_momentum_experiment():
    betas = [0.0, 0.5, 0.9, 0.99]
    momentum_results = []
    plt.figure(figsize=(10, 6))

    for beta in betas:
        model = PerceptronWithMomentum(seed=42)
        losses = model.fit(
            X_train,
            Y_train,
            X_test,
            Y_test,
            epochs=100,
            lr=0.1,
            batch_size=32,
            momentum=beta,
        )
        accuracy = np.mean(model.predict(X_test) == Y_test)
        momentum_results.append({"Beta": beta, "Test_Accuracy": f"{accuracy:.4f}"})
        plt.plot(losses, label=f"beta={beta}")

    plt.title("Impact of Momentum on Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.show()

    momentum_results_df = pd.DataFrame(momentum_results)
    display(momentum_results_df)
    return momentum_results_df


def cross_validate(X, y, lrs, batch_sizes, k=5, epochs=50):
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    results = []

    for lr in lrs:
        for batch_size in batch_sizes:
            fold_accuracies = []
            for i in range(k):
                val_idx = indices[i * fold_size: (i + 1) * fold_size]
                train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                model = Perceptron(seed=42)

                for _ in range(epochs):
                    shuffler = np.random.permutation(len(X_tr))
                    X_shuffled, y_shuffled = X_tr[shuffler], y_tr[shuffler]
                    current_bs = min(batch_size, len(X_tr))
                    for j in range(0, len(X_tr), current_bs):
                        X_batch = X_shuffled[j: j + current_bs]
                        y_batch = y_shuffled[j: j + current_bs]
                        m = X_batch.shape[0]
                        error = model.forward(X_batch) - y_batch
                        model.w -= lr * (X_batch.T @ error) / m
                        model.b -= lr * np.sum(error) / m

                fold_accuracies.append(np.mean(model.predict(X_val) == y_val))

            results.append(
                {
                    "lr": lr,
                    "batch_size": batch_size,
                    "mean_accuracy": np.mean(fold_accuracies),
                    "std_accuracy": np.std(fold_accuracies),
                }
            )

    return pd.DataFrame(results)


def run_hyperparameter_selection():
    cv_results_df = cross_validate(
        X_train,
        Y_train,
        lrs=[0.01, 0.1, 0.5],
        batch_sizes=[16, 32, 64],
    )
    display(cv_results_df)
    best_params = cv_results_df.loc[cv_results_df["mean_accuracy"].idxmax()]
    print(f"\nЛучшие параметры: LR={best_params['lr']}, BatchSize={int(best_params['batch_size'])}")

    final_model = Perceptron(seed=42)
    final_model.fit(
        X_train,
        Y_train,
        X_test,
        Y_test,
        epochs=100,
        lr=best_params["lr"],
        batch_size=int(best_params["batch_size"]),
    )
    final_test_accuracy = np.mean(final_model.predict(X_test) == Y_test)
    print(f"\nЛучший accuracy: {final_test_accuracy:.4f}")
    return cv_results_df, final_model


def run_metrics_analysis(final_model):
    y_pred_test = final_model.predict(X_test)
    y_probs_test = final_model.forward(X_test)
    accuracy = np.mean(Y_test == y_pred_test)
    precision = compute_precision(Y_test, y_pred_test)
    recall = compute_recall(Y_test, y_pred_test)
    f1 = compute_f1(Y_test, y_pred_test)
    fpr, tpr, auc_val = compute_roc_auc(Y_test, y_probs_test)

    metrics = pd.DataFrame(
        [
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": auc_val,
            }
        ]
    )
    display(metrics)
    plot_roc_and_errors(
        final_model,
        X_test,
        Y_test,
        fpr,
        tpr,
        auc_val,
        accuracy,
        precision,
        recall,
    )
    return metrics


def main():
    run_parameter_experiments()
    run_synthetic_data_experiment()
    run_loss_and_regularization_experiment()
    _, final_model = run_hyperparameter_selection()
    run_metrics_analysis(final_model)
    run_momentum_experiment()


if __name__ == "__main__":
    main()
