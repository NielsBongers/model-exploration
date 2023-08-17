import numpy
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_recall_fscore_support,
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import cross_validate


def logistic_regression(X, y, scoring, K=5):
    clf = LogisticRegression(max_iter=500)
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=K, n_jobs=10)
    return cv_results


def nearest_neighbors(X, y, scoring, K=5):
    clf = KNeighborsClassifier(algorithm="ball_tree")
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=K, n_jobs=10)
    return cv_results


def support_vector_linear(X, y, scoring, K=5):
    clf = SVC(kernel="linear")
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=K, n_jobs=10)
    return cv_results


def support_vector_rbf(X, y, scoring, K=5):
    clf = SVC(kernel="rbf")
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=K, n_jobs=10)
    return cv_results


def support_vector_sigmoid(X, y, scoring, K=5):
    clf = SVC(kernel="sigmoid")
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=K, n_jobs=10)
    return cv_results


def random_forest(X, y, scoring, K=5):
    clf = RandomForestClassifier()
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=K, n_jobs=10)
    return cv_results


def xgboost_classifier(X, y, scoring, K=5):
    clf = XGBClassifier()
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=K, n_jobs=10)
    return cv_results


def get_model_exploration(
    X: numpy.array,
    y: numpy.array,
    title="Exploration",
    K=5,
    save_figure=True,
    show_figure=True,
    show_output=True,
    latex_output=False,
) -> pd.DataFrame:
    """Runs K-fold cross validation (default=5) over a number of different machine learning options:
            Logistic regression,
            Nearest neighbor,
            Support vector machines with a linear, radial basis function and sigmoid kernel,
            Random forest,
            XGBoost,

       Returns scoring for accuracy, precision, recall and F1 in a DataFrame, and optionally displays and/or saves an image.

    Args:
        X (numpy.array): Training values.
        y (numpy.array): Labels associated with training values.
        title (str, optional): Title for the figure. Defaults to "Exploration".
        K (int, optional): Number of cross-validation folds to run. Minimum value is 2. Defaults to 5.
        save_figure (bool, optional): Save the figure to Figures/. Defaults to True.
            show_figure (bool, optional): Display the figure. Defaults to True.
        show_output (bool, optional): Show output table. Defaults to True.
        latex_output (bool, optional): Show LaTeX-formatted output table. Defaults to False.

    Returns:
        pd.DataFrame: Frame with the evaluation metrics.
    """

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }

    classifier_set = {
        "Logistic regression": logistic_regression,
        "Nearest neighbor": nearest_neighbors,
        "Linear SVM": support_vector_linear,
        "RBF SVM": support_vector_rbf,
        "Sigmoid SVM": support_vector_sigmoid,
        "Random forest": random_forest,
        "XGBoost": xgboost_classifier,
    }

    group_list = []

    for index, (key) in enumerate(classifier_set.keys()):
        score = classifier_set[key](X.values, y.values.ravel(), scoring, K=K)

        accuracy = score["test_accuracy"].mean()
        precision = score["test_precision"].mean()
        recall = score["test_recall"].mean()
        f1 = score["test_f1_score"].mean()

        group_list.append([accuracy, precision, recall, f1])

    df = pd.DataFrame(group_list, index=classifier_set.keys())
    df.columns = ["Accuracy", "Precision", "Recall", "F1"]

    df.plot.bar()
    plt.xticks(rotation=45)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), ncol=1)
    plt.title(title)
    plt.tight_layout()

    if save_figure:
        Path("Figures").mkdir(exist_ok=True)
        plt.savefig(Path("Figures", title + ".png"), dpi=300)
    if show_figure:
        plt.show()

    if show_output:
        print(df.round(decimals=3))
    if latex_output:
        print(df.style.to_latex())

    return df


def get_PCA(
    df: pd.DataFrame,
    selected_features=None,
    title="PCA",
    split_feature=None,
    n_components=2,
    show_figure=True,
    save_figure=True,
):
    """Creates a PCA from the current dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        selected_features (list, optional): Features to do the PCA on. Only works for numerical values. Defaults to None.
        split_feature (string, optional): Features plot separately (for example, multiple different classes). Defaults to None.
        n_components (int, optional): Number of PCA components. Defaults to 2.
        show_figure (bool, optional): Plot and show the PCA. Defaults to True.
        save_figure (bool, optional): Plot and save the PCA. Defaults to True.

    Raises:
        KeyError: Raised if the specified split feature is not in the DataFrame's columns.
        ValueError: Raised if the number of components is not two when plotting the PCA.
    """

    pca_decomposition = PCA(n_components=n_components)

    try:
        pca_decomposition.fit(df[selected_features] if selected_features else df)
    except Exception as e:
        print(f"Error during PCA decomposition: {e}")

    print(f"Explained variance ratio: {pca_decomposition.explained_variance_ratio_}")

    try:
        if split_feature and split_feature not in df.columns:
            raise KeyError(
                f"Split feature {split_feature} not found in DataFrame. Options are: {', '.join(list(df.columns))}"
            )
    except Exception as e:
        print(f"Exception: {e}")

    if n_components == 2 and (show_figure or save_figure):
        if split_feature:
            unique_classes = df[split_feature].unique()

            for unique_class in unique_classes:
                selected_df = df[df[split_feature] == unique_class]
                selected_pca = pca_decomposition.transform(
                    selected_df[selected_features] if selected_features else selected_df
                )
                plt.scatter(selected_pca[:, 0], selected_pca[:, 1], alpha=0.5)

            plt.legend(
                [
                    str(split_feature) + ": " + str(selected_class)
                    for selected_class in unique_classes
                ]
            )

        else:
            selected_pca = pca_decomposition.transform(
                df[selected_features] if selected_features else df
            )

            plt.scatter(selected_pca[:, 0], selected_pca[:, 1], alpha=0.5)

        plt.title(title)
        plt.tight_layout()

        if save_figure:
            Path("Figures").mkdir(exist_ok=True)
            destination_path = Path("Figures", title + ".png")
            plt.savefig(destination_path, dpi=300)
            print(f"Saved figure to {destination_path}")
        if show_figure:
            plt.show()

    try:
        if n_components != 2 and (show_figure or save_figure):
            raise ValueError(
                f"Plotting is possible/interesting mostly for n_components = 2. Currently selected: {n_components}"
            )

    except ValueError as e:
        print(f"ValueError: {e}")
