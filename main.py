# Evon Troy Alexander
# CAP4779
# Professor Zad
# April 15, 2024

from sklearn.decomposition import PCA
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


# Load and prepare data
def load_and_prepare_data():
    iris = datasets.load_iris()
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(iris.data)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.15, random_state=23)
    return X_train, X_test, y_train, y_test


# Plotting function
def plot_decision_boundaries(X_train, y_train, X_test, y_test, models, titles, subplot_dims):
    fig, sub = plt.subplots(*subplot_dims, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for clf, title, ax in zip(models, titles, sub.flatten()):
        clf.fit(X_train, y_train)
        DecisionBoundaryDisplay.from_estimator(
            clf, X_train, response_method="predict",
            cmap=plt.cm.coolwarm, alpha=0.8, ax=ax, xlabel='PCA.f0', ylabel='PCA.f1'
        )
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        score = clf.score(X_test, y_test)
        ax.set_title(f"{title}\nScore: {score:.2%}")
    plt.show()


# steps 2 to 5
def execute_steps(X_train, y_train, X_test, y_test):
    steps_config = {
        "Step 2": {
            "models": [svm.SVC(kernel='poly', degree=d, gamma=g, C=1) for d in [3, 5, 7] for g in [0.3, 0.5]],
            "titles": [f"SVM Poly Degree {d}, Gamma {g}" for d in [3, 5, 7] for g in [0.3, 0.5]],
            "subplot_dims": (2, 3)
        },
        "Step 3": {
            "models": [GaussianNB(var_smoothing=10 ** (-i)) for i in range(-1, 5)],
            "titles": [f"GaussianNB var_smoothing=1e{i}" for i in range(-1, 5)],
            "subplot_dims": (2, 3)
        },
        "Step 4": {
            "models": [svm.SVC(kernel=k, gamma=g, C=1) for k in ['sigmoid', 'rbf'] for g in [0.1, 1, 100]],
            "titles": [f"SVM {k} Gamma {g}" for k in ['sigmoid', 'rbf'] for g in [0.1, 1, 100]],
            "subplot_dims": (2, 3)
        },
        "Step 5": {
            "models": [
                MLPClassifier(solver=s, activation=a, hidden_layer_sizes=l, alpha=0.0001, random_state=23)
                for s in ['adam', 'lbfgs'] for a in ['logistic', 'relu'] for l in [(30, 30), (10, 5)]
            ],
            "titles": [
                f"MLP {s} {a} layers {l}" for s in ['adam', 'lbfgs'] for a in ['logistic', 'relu'] for l in
                [(30, 30), (10, 5)]
            ],
            "subplot_dims": (2, 4)
        }
    }

    # Execute each step
    for step, config in steps_config.items():
        print(f"\n{step} Results:")
        plot_decision_boundaries(X_train, y_train, X_test, y_test, config['models'], config['titles'],
                                 config['subplot_dims'])


# Main execution
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    execute_steps(X_train, y_train, X_test, y_test)
