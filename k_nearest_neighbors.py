import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    training_data = pd.read_csv('training.csv/training.csv')
    y = training_data['label']
    X = training_data.drop(['label'], axis=1)

    tuned_parameters = [
        {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10],
         "weights":['uniform', 'distance']}
    ]

    knn_clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring="accuracy", n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    knn_clf.fit(X_train, y_train)

    # use the best param to print out the result of cross validation
    scores = cross_val_score(knn_clf.best_estimator_, X_test, y_test, scoring="accuracy", cv=5)
    print(scores)
    plt.plot(range(1, 6, 1), scores)
    plt.xlabel('Folds')
    plt.ylabel('Accuracy')
    plt.ylim((0.7, 1.0))
    plt.grid()
    plt.show()
    print("average accuracy of five folds: ", np.mean(scores))

    print("now use the entire training data to train...")
    clf = KNeighborsClassifier()
    clf.set_params(**knn_clf.best_params_)
    clf.fit(X, y)

    res = clf.predict(pd.read_csv('testing.csv/testing.csv'))

    print()
    print("predicted the testing data and saving the result into csv")
    pd.DataFrame(res).to_csv('KNN_pred.csv', index=False, encoding="utf-8")

    # uncomment the following lines to show the accuracy of this model

    # y_Test = pd.read_csv("PUT THE TESTING LABEL FILE HERE")
    # print("The performance of this model on testing dataset is:", accuracy_score(y_Test, res))
