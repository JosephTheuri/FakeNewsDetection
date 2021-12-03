import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import model_selection
from xgboost import XGBClassifier
import pandas as pd

class TrainModel():
    def __init__(self, X_train, X_test, y_train, y_test, columns):
        self.train = X_train
        self.y_train = y_train
        self.test = X_test
        self.y_test = y_test
        self.columns = columns

    def train_results(self):
        models = []
        dfs = []

        # Create a list of possible test models
        models.append(("PassiveAggressive", PassiveAggressiveClassifier(random_state=0)))
        models.append(("LogisticRegression", LogisticRegression(random_state=0)))
        models.append(("SVC", SVC(random_state=0)))
        models.append(("XGBoost", XGBClassifier(use_label_encoder=True)))
        models.append(("KNeighbors", KNeighborsClassifier()))
        models.append(("DecisionTree", DecisionTreeClassifier(random_state=0)))
        models.append(("RandomForest", RandomForestClassifier(n_estimators=10, max_depth=20, min_samples_split=50, random_state=0)))

        # Train Models on training data to find best model
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        target_names = ['Reliable', 'Unreliable']
        for name, model in models:
            print(name)
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
            cv_results = model_selection.cross_validate(model, self.train, self.y_train, cv=kfold, scoring=scoring)
            cv_mean = pd.DataFrame(cv_results).mean()
            this_df = cv_mean
            this_df['model'] = name
            dfs.append(this_df)
            model.fit(self.train, self.y_train)
            joblib.dump(model, 'model\{}.pkl'.format(name))
            if name == "RandomForest":
                feat_importances = pd.Series(model.feature_importances_,
                                            index= self.columns
                                            ).sort_values(ascending=False)
                print(feat_importances)
                feat_importances.to_csv('feature_importance.csv')


        train = pd.concat(dfs, ignore_index=True, axis=1).transpose().sort_values('test_roc_auc', ascending=False).reset_index()
        top = train['model'][0]
        train.to_csv('src\\modelling\\train_results.csv', index=False)

        # Test Final Model Model
        top_model = [y for x,y in models if x == top][0]  # Select top model
        print("Top Model:", top_model)


        clf = joblib.load("model\{}.pkl".format(top))
        y_pred = clf.predict(self.test)
        report = classification_report(self.y_test, y_pred, target_names=target_names, output_dict=True)
        print(classification_report(self.y_test, y_pred, target_names=target_names))
        final = pd.DataFrame(report).transpose()

        final.to_csv('src\\modelling\\final_results.csv', index=False)
            
            # results.append(report)
            # names.append(name)
        
        return top
