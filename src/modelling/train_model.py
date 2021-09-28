from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import model_selection
import pandas as pd

class TrainModel():
    def __init__(self, tfidf_train, tfidf_test, y_train, y_test):
        self.train = tfidf_train
        self.y_train = y_train
        self.test = tfidf_test
        self.y_test = y_test

    def train_results(self):
        models = []
        dfs = []

        models.append(("PassiveAggressive", PassiveAggressiveClassifier(max_iter=50)))
        models.append(("LogisticRegression", LogisticRegression()))
        # models.append(("SVC", SVC()))
        # models.append(("KNeighbors", KNeighborsClassifier()))
        # models.append(("DecisionTree", DecisionTreeClassifier()))
        models.append(("RandomForest", RandomForestClassifier()))

        results = []
        names = []
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        target_names = ['Real', 'Fake']
        for name, model in models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
            cv_results = model_selection.cross_validate(model, self.train, self.y_train, cv=kfold, scoring=scoring)
            clf = model.fit(self.train, self.y_train)
            y_pred = clf.predict(self.test)
            print(name)
            print(classification_report(self.y_test, y_pred, target_names=target_names))
            
            results.append(cv_results)
            names.append(name)
            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            dfs.append(this_df)
        
        final = pd.concat(dfs, ignore_index=True)
        final.to_csv('src\\modelling\\train_results.csv', index=False)
        return final
