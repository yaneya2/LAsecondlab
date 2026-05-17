import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dataset import X_train, X_test, Y_train, Y_test
from perceptron import Perceptron

try:
    from IPython.display import display
except ImportError:
    display = print


results = []

lr = [0.001, 0.01, 0.5, 1.0, 2.0, 5.0]

plt.figure(figsize=(10, 5))

for cur_lr in lr:
    print(f"current learning rate = {cur_lr}")
    model = Perceptron(seed=42)
    train_losses, val_losses = model.fit(
        X_train, Y_train,
        X_test, Y_test,
        epochs=100,
        lr=cur_lr,
        batch_size=32
    )

    results.append({'Exp': 'learning rate', 'param': cur_lr, 'accuracy': np.mean(model.predict(X_test) == Y_test), 'final_Train_Loss': train_losses[-1], 'final_Val_Loss': val_losses[-1]})

    print(50 * "-")
    plt.plot(val_losses, label=f"rl={cur_lr}")

plt.title(f"Experiment: learning_rate")
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.legend()
plt.grid(True)
plt.show()

bs = [1, 16, 64, 256]

plt.figure(figsize=(10, 5))

for cur_bs in bs:
    print(f"current batch size= {cur_bs}")
    model = Perceptron(seed=42)
    train_losses, val_losses = model.fit(
        X_train, Y_train,
        X_test, Y_test,
        epochs=100,
        lr=0.1,
        batch_size=cur_bs
    )

    results.append({'Exp': 'batch', 'param': cur_bs, 'accuracy': np.mean(model.predict(X_test) == Y_test), 'final_Train_Loss': train_losses[-1], 'final_Val_Loss': val_losses[-1]})

    print(50 * "-")
    plt.plot(val_losses, label=f"batch size={cur_bs}")

plt.title(f"Experiment: batch_size")
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.legend()
plt.grid(True)
plt.show()

i_type = [0, 1, 10]
name_type = {0: 'zero', 1: 'small', 10: 'large'}

plt.figure(figsize=(10, 5))

for cur_type in i_type:
    print(f"current type= {cur_type}")
    model = Perceptron(seed=42)
    model.w = np.random.randn(2) * cur_type
    train_losses, val_losses = model.fit(
        X_train, Y_train,
        X_test, Y_test,
        epochs=100,
        lr=0.1,
        batch_size=32
    )

    results.append({'Exp': 'init_type', 'param': name_type[cur_type], 'accuracy': np.mean(model.predict(X_test) == Y_test), 'final_Train_Loss': train_losses[-1], 'final_Val_Loss': val_losses[-1]})

    print(50 * "-")
    plt.plot(val_losses, label=f"type={name_type[cur_type]}")

plt.title(f"Experiment: init_type")
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
plt.legend()
plt.grid(True)
plt.show()

df = pd.DataFrame(results)

display(df)


def generate_linear(n=500, centers=[(-2, -2), (2, 2)], cov=[[1, 0], [0, 1]]):
    n_half = n // 2
    c0 = np.random.multivariate_normal(centers[0], cov, n_half)
    c1 = np.random.multivariate_normal(centers[1], cov, n - n_half)
    X = np.vstack([c0, c1])
    y = np.hstack([np.zeros(n_half), np.ones(n - n_half)])
    return X, y


def generate_xor(n=500, spread=0.3):
    n_q = n // 4

    q1 = np.random.normal([1, 1], spread, (n_q, 2))
    q2 = np.random.normal([-1, -1], spread, (n_q, 2))
    q3 = np.random.normal([-1, 1], spread, (n_q, 2))
    q4 = np.random.normal([1, -1], spread, (n - 3 * n_q, 2))
    X = np.vstack([q1, q2, q3, q4])
    y = np.hstack([np.zeros(2 * n_q), np.ones(n - 2 * n_q)])
    return X, y


def generate_circle(n=500, inner_r=1.0, outer_r=2.5):
    n_half = n // 2

    r_in = np.sqrt(np.random.uniform(0, inner_r ** 2, n_half))
    theta_in = np.random.uniform(0, 2 * np.pi, n_half)
    c1 = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])

    r_out = np.sqrt(np.random.uniform(inner_r ** 2 + 0.5, outer_r ** 2, n - n_half))
    theta_out = np.random.uniform(0, 2 * np.pi, n - n_half)
    c0 = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    X = np.vstack([c1, c0])
    y = np.hstack([np.ones(n_half), np.zeros(n - n_half)])
    return X, y


def plot_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=20)
    acc = np.mean(model.predict(X) == y)
    ax.set_title(f"{title}\nAcc: {acc:.2f}")


datasets = {
    "Linear": generate_linear(),
    "XOR": generate_xor(),
    "Circle": generate_circle()
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, (X, y)) in enumerate(datasets.items()):
    indices = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    model = Perceptron(seed=42)
    model.fit(X[train_idx], y[train_idx], X[test_idx], y[test_idx], epochs=100, lr=0.1, batch_size=16)

    plot_boundary(axes[i], model, X, y, name)

plt.tight_layout()
plt.show()


def cross_validate(X, y, lrs, batch_sizes, k=5, epochs=50):
    n_samples = X.shape[0]
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    results = []

    for lr in lrs:
        for bs in batch_sizes:
            fold_accuracies = []

            for i in range(k):
                val_idx = indices[i * fold_size: (i + 1) * fold_size]
                train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]

                model_cv = Perceptron(seed=42)

                for epoch in range(epochs):
                    shuffler = np.random.permutation(len(X_tr))
                    X_s, y_s = X_tr[shuffler], y_tr[shuffler]
                    current_bs = min(bs, len(X_tr))

                    for j in range(0, len(X_tr), current_bs):
                        X_b = X_s[j: j + current_bs]
                        y_b = y_s[j: j + current_bs]
                        m = X_b.shape[0]

                        y_hat = model_cv.forward(X_b)
                        err = y_hat - y_b
                        dw = (1 / m) * (X_b.T @ err)
                        db = (1 / m) * np.sum(err)
                        model_cv.w -= lr * dw
                        model_cv.b -= lr * db

                acc = np.mean(model_cv.predict(X_va) == y_va)
                fold_accuracies.append(acc)

            results.append({
                'lr': lr,
                'batch_size': bs,
                'mean_accuracy': np.mean(fold_accuracies),
                'std_accuracy': np.std(fold_accuracies)
            })

    return pd.DataFrame(results)


lr_list = [0.01, 0.1, 0.5]
bs_list = [16, 32, 64]

cv_results_df = cross_validate(X_train, Y_train, lr_list, bs_list)
display(cv_results_df)

best_params = cv_results_df.loc[cv_results_df['mean_accuracy'].idxmax()]
print(f"\nЛучшие параметры: LR={best_params['lr']}, BatchSize={int(best_params['batch_size'])}")

final_model = Perceptron(seed=42)
final_model.fit(
    X_train, Y_train, X_test, Y_test,
    epochs=100,
    lr=best_params['lr'],
    batch_size=int(best_params['batch_size'])
)

final_test_acc = np.mean(final_model.predict(X_test) == Y_test)
print(f"\nЛучший accuarcy: {final_test_acc:.4f}")
