import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import warnings
import seaborn as sns
import pickle

df = pd.read_csv('phishing.csv')


df = df.drop(['Index'],axis = 1)
X = df.drop(["class"],axis =1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model_name = []
accuracy = []
f1_score = []
recall = []
precision = []

def GetMetrics(model, acc,f1,rec,pre): #gets metrics from all models for final analysis
  model_name.append(model)
  accuracy.append(round(acc, 3))
  f1_score.append(round(f1, 3))
  recall.append(round(rec, 3))
  precision.append(round(pre, 3))

def logistic_regression(X_train,y_train,X_test,y_test):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression()
    log.fit(X_train,y_train)
    y_train_log = log.predict(X_train)
    y_test_log = log.predict(X_test)

    acc_train_log = metrics.accuracy_score(y_train,y_train_log)
    acc_test_log = metrics.accuracy_score(y_test,y_test_log)
    print("Logistic Regression : Accuracy on training Data: {:.3f}".format(acc_train_log))
    print("Logistic Regression : Accuracy on test Data: {:.3f}".format(acc_test_log))
    print()

    f1_score_train_log = metrics.f1_score(y_train,y_train_log)
    f1_score_test_log = metrics.f1_score(y_test,y_test_log)
    print("Logistic Regression : f1_score on training Data: {:.3f}".format(f1_score_train_log))
    print("Logistic Regression : f1_score on test Data: {:.3f}".format(f1_score_test_log))
    print()

    recall_score_train_log = metrics.recall_score(y_train,y_train_log)
    recall_score_test_log = metrics.recall_score(y_test,y_test_log)
    print("Logistic Regression : Recall on training Data: {:.3f}".format(recall_score_train_log))
    print("Logistic Regression : Recall on test Data: {:.3f}".format(recall_score_test_log))
    print()

    precision_score_train_log = metrics.precision_score(y_train,y_train_log)
    precision_score_test_log = metrics.precision_score(y_test,y_test_log)
    print("Logistic Regression : precision on training Data: {:.3f}".format(precision_score_train_log))
    print("Logistic Regression : precision on test Data: {:.3f}".format(precision_score_test_log))

    con  = confusion_matrix(y_train,y_train_log)
    sns.heatmap(con,annot=True,fmt='.2f')

    print(metrics.classification_report(y_test, y_test_log))

    GetMetrics('Logistic Regression',acc_test_log,f1_score_test_log,
                 recall_score_train_log,precision_score_train_log)

def knn(X_train,y_train,X_test,y_test):

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    y_train_knn = knn.predict(X_train)
    y_test_knn = knn.predict(X_test)
    acc_train_knn = metrics.accuracy_score(y_train,y_train_knn)
    acc_test_knn = metrics.accuracy_score(y_test,y_test_knn)
    print("K-Nearest Neighbors : Accuracy on training Data: {:.3f}".format(acc_train_knn))
    print("K-Nearest Neighbors : Accuracy on test Data: {:.3f}".format(acc_test_knn))
    print()

    f1_score_train_knn = metrics.f1_score(y_train,y_train_knn)
    f1_score_test_knn = metrics.f1_score(y_test,y_test_knn)
    print("K-Nearest Neighbors : f1_score on training Data: {:.3f}".format(f1_score_train_knn))
    print("K-Nearest Neighbors : f1_score on test Data: {:.3f}".format(f1_score_test_knn))
    print()

    recall_score_train_knn = metrics.recall_score(y_train,y_train_knn)
    recall_score_test_knn = metrics.recall_score(y_test,y_test_knn)
    print("K-Nearest Neighbors : Recall on training Data: {:.3f}".format(recall_score_train_knn))
    print("K-Nearest Neighbors : Recall on test Data: {:.3f}".format(recall_score_test_knn))
    print()

    precision_score_train_knn = metrics.precision_score(y_train,y_train_knn)
    precision_score_test_knn = metrics.precision_score(y_test,y_test_knn)
    print("K-Nearest Neighbors : precision on training Data: {:.3f}".format(precision_score_train_knn))
    print("K-Nearest Neighbors : precision on test Data: {:.3f}".format(precision_score_test_knn))

    print(metrics.classification_report(y_test, y_test_knn))
    GetMetrics('K-Nearest Neighbors', acc_test_knn, f1_score_test_knn,
               recall_score_train_knn, precision_score_train_knn)

def svm(X_train,y_train,X_test,y_test):

    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    param_grid = {'gamma': [0.1],'kernel': ['rbf','linear']}
    svc = GridSearchCV(SVC(), param_grid)
    svc.fit(X_train, y_train)

    y_train_svc = svc.predict(X_train)
    y_test_svc = svc.predict(X_test)
    acc_train_svc = metrics.accuracy_score(y_train,y_train_svc)
    acc_test_svc = metrics.accuracy_score(y_test,y_test_svc)
    print("Support Vector Machine : Accuracy on training Data: {:.3f}".format(acc_train_svc))
    print("Support Vector Machine : Accuracy on test Data: {:.3f}".format(acc_test_svc))
    print()

    f1_score_train_svc = metrics.f1_score(y_train,y_train_svc)
    f1_score_test_svc = metrics.f1_score(y_test,y_test_svc)
    print("Support Vector Machine : f1_score on training Data: {:.3f}".format(f1_score_train_svc))
    print("Support Vector Machine : f1_score on test Data: {:.3f}".format(f1_score_test_svc))
    print()

    recall_score_train_svc = metrics.recall_score(y_train,y_train_svc)
    recall_score_test_svc = metrics.recall_score(y_test,y_test_svc)
    print("Support Vector Machine : Recall on training Data: {:.3f}".format(recall_score_train_svc))
    print("Support Vector Machine : Recall on test Data: {:.3f}".format(recall_score_test_svc))
    print()

    precision_score_train_svc = metrics.precision_score(y_train,y_train_svc)
    precision_score_test_svc = metrics.precision_score(y_test,y_test_svc)
    print("Support Vector Machine : precision on training Data: {:.3f}".format(precision_score_train_svc))
    print("Support Vector Machine : precision on test Data: {:.3f}".format(precision_score_test_svc))

    con  = confusion_matrix(y_train,y_train_svc)
    sns.heatmap(con,annot=True,fmt='.2f')
    print(metrics.classification_report(y_test, y_test_svc))
    GetMetrics('Support Vector Machine', acc_test_svc, f1_score_test_svc,
               recall_score_train_svc, precision_score_train_svc)

def random_forest(X_train,y_train,X_test,y_test):

    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10)
    forest.fit(X_train,y_train)
    y_train_forest = forest.predict(X_train)
    y_test_forest = forest.predict(X_test)
    acc_train_forest = metrics.accuracy_score(y_train,y_train_forest)
    acc_test_forest = metrics.accuracy_score(y_test,y_test_forest)
    print("Random Forest : Accuracy on training Data: {:.3f}".format(acc_train_forest))
    print("Random Forest : Accuracy on test Data: {:.3f}".format(acc_test_forest))
    print()

    f1_score_train_forest = metrics.f1_score(y_train,y_train_forest)
    f1_score_test_forest = metrics.f1_score(y_test,y_test_forest)
    print("Random Forest : f1_score on training Data: {:.3f}".format(f1_score_train_forest))
    print("Random Forest : f1_score on test Data: {:.3f}".format(f1_score_test_forest))
    print()

    recall_score_train_forest = metrics.recall_score(y_train,y_train_forest)
    recall_score_test_forest = metrics.recall_score(y_test,y_test_forest)
    print("Random Forest : Recall on training Data: {:.3f}".format(recall_score_train_forest))
    print("Random Forest : Recall on test Data: {:.3f}".format(recall_score_test_forest))
    print()

    precision_score_train_forest = metrics.precision_score(y_train,y_train_forest)
    precision_score_test_forest = metrics.precision_score(y_test,y_test_forest)
    print("Random Forest : precision on training Data: {:.3f}".format(precision_score_train_forest))
    print("Random Forest : precision on test Data: {:.3f}".format(precision_score_test_forest))

    con  = confusion_matrix(y_train,y_train_forest)
    sns.heatmap(con,annot=True,fmt='.2f')
    print(metrics.classification_report(y_test, y_test_forest))
    GetMetrics('Random Forest',acc_test_forest,f1_score_test_forest,
                 recall_score_train_forest,precision_score_train_forest)
    with open('random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(forest, model_file)


print("Four classification models")

logistic_regression(X_train,y_train,X_test,y_test)
knn(X_train,y_train,X_test,y_test)
svm(X_train,y_train,X_test,y_test)
random_forest(X_train,y_train,X_test,y_test)

final_observations = pd.DataFrame({ 'ML Model' : model_name,
                        'Accuracy' : accuracy,
                        'f1_score' : f1_score,
                        'Recall'   : recall,
                        'Precision': precision,
                      })
sorted_final_observations=final_observations.sort_values(by=['Accuracy', 'f1_score'],ascending=False).reset_index(drop=True)
print(sorted_final_observations)

# Save the trained model to a pickle file
# Load the pre-trained model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Prepare link data (you need to implement this)
link_data = preprocess_link('https://example.com')  # Replace with your preprocessing function

# Make predictions
prediction = model.predict(link_data)

# Interpret the prediction
if prediction == 1:
    print("The link is classified as phishing.")
else:
    print("The link is classified as legitimate.")
