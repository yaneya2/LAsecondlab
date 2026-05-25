import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_notebook_dataset():
    """Create the base split exactly as it is used in the notebook."""
    x, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
    )

    scaler = StandardScaler()
    scaler.fit(x)

    X_train, X_test, Y_train, Y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )
    return x, y, scaler, X_train, X_test, Y_train, Y_test


def create_standardized_dataset():
    """Create the base split with train-only Z-score scaling required by the task."""
    x, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return x, y, scaler, X_train, X_test, Y_train, Y_test


def _apply_label_noise(y, noise=0.0):
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be between 0 and 1")
    y_noisy = y.copy()
    if noise > 0.0:
        flips = np.random.random(y.shape[0]) < noise
        y_noisy[flips] = 1 - y_noisy[flips]
    return y_noisy


def generate_linear(n=500, centers=((-2, -2), (2, 2)), cov=((1, 0), (0, 1)), noise=0.0):
    n_half = n // 2
    c0 = np.random.multivariate_normal(centers[0], cov, n_half)
    c1 = np.random.multivariate_normal(centers[1], cov, n - n_half)
    X = np.vstack([c0, c1])
    y = np.hstack([np.zeros(n_half), np.ones(n - n_half)])
    return X, _apply_label_noise(y, noise)


def generate_xor(n=500, spread=0.3, noise=0.0):
    n_q = n // 4
    q1 = np.random.normal([1, 1], spread, (n_q, 2))
    q2 = np.random.normal([-1, -1], spread, (n_q, 2))
    q3 = np.random.normal([-1, 1], spread, (n_q, 2))
    q4 = np.random.normal([1, -1], spread, (n - 3 * n_q, 2))
    X = np.vstack([q1, q2, q3, q4])
    y = np.hstack([np.zeros(2 * n_q), np.ones(n - 2 * n_q)])
    return X, _apply_label_noise(y, noise)


def generate_circle(n=500, inner_r=1.0, outer_r=2.5, noise=0.0):
    n_half = n // 2
    r_in = np.sqrt(np.random.uniform(0, inner_r**2, n_half))
    theta_in = np.random.uniform(0, 2 * np.pi, n_half)
    c1 = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])

    r_out = np.sqrt(np.random.uniform(inner_r**2 + 0.5, outer_r**2, n - n_half))
    theta_out = np.random.uniform(0, 2 * np.pi, n - n_half)
    c0 = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    X = np.vstack([c1, c0])
    y = np.hstack([np.ones(n_half), np.zeros(n - n_half)])
    return X, _apply_label_noise(y, noise)


x, y, scaler, X_train, X_test, Y_train, Y_test = create_notebook_dataset()
