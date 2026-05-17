from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# создание датасета
x, y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
    n_clusters_per_class=1
)

# нормализация
scaler = StandardScaler()

scaler.fit(x)

# разбиение на выборки
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
