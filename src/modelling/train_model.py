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

        # Create a list of possible test models
        models.append(("PassiveAggressive", PassiveAggressiveClassifier(max_iter=50)))
        models.append(("LogisticRegression", LogisticRegression()))
        models.append(("SVC", SVC()))
        models.append(("KNeighbors", KNeighborsClassifier()))
        models.append(("DecisionTree", DecisionTreeClassifier()))
        models.append(("RandomForest", RandomForestClassifier()))

        # Train Models on training data to find best model
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        target_names = ['Real', 'Fake']
        for name, model in models:
            print(name)
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
            cv_results = model_selection.cross_validate(model, self.train, self.y_train, cv=kfold, scoring=scoring)
            cv_mean = pd.DataFrame(cv_results).mean()
            this_df = cv_mean
            this_df['model'] = name
            dfs.append(this_df)

        train = pd.concat(dfs, ignore_index=True, axis=1).transpose()
        train.to_csv('src\\modelling\\train_results.csv', index=False)

        # Test Final Model Model
        top_model = [y for x,y in models if x == train['model'][0]][0]
        print("Top Model:", top_model)
        clf = top_model.fit(self.train, self.y_train)
        y_pred = clf.predict(self.test)
        report = classification_report(self.y_test, y_pred, target_names=target_names, output_dict=True)
        final = pd.DataFrame(report).transpose()

        final.to_csv('src\\modelling\\final_results.csv', index=False)
            
            # results.append(report)
            # names.append(name)
        
        return report
