import numpy 
import pandas as pd 
import matplotlib.pyplot as plt 

from pathlib import Path 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_recall_fscore_support, make_scorer, accuracy_score, precision_score, recall_score, f1_score
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


def explore(X: numpy.array, y: numpy.array, title="Exploration", K=5, save_figure=True, show_figure=True, show_output=True, latex_output=False) -> pd.DataFrame: 
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
        K (int, optional): Number of cross-validation folds to run. Defaults to 5.
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
        "f1_score": make_scorer(f1_score)}
    
    classifier_set = {
        "Logistic regression": logistic_regression, 
        "Nearest neighbor": nearest_neighbors, 
        "Linear SVM": support_vector_linear, 
        "RBF SVM": support_vector_rbf, 
        "Sigmoid SVM": support_vector_sigmoid, 
        "Random forest": random_forest, 
        "XGBoost": xgboost_classifier
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
    plt.legend(loc="upper right", bbox_to_anchor=(1.3,1.05), ncol=1)
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
