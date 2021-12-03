import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.metrics import classification_report, accuracy_score

class OptimizeRF():
    def __init__(self, X_train, X_test, y_train, y_test, columns):
        self.train = X_train
        self.y_train = y_train
        self.test = X_test
        self.y_test = y_test
        self.columns = columns

    def stratified_kfold_score(self, clf):
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        cv_results = cross_validate(clf, self.train, self.y_train, cv=kfold, scoring='accuracy')
        cv_mean = pd.DataFrame(cv_results)['test_score'].mean()
        # print(cv_mean)

        return cv_mean

    def bo_params_rf(self, max_samples, n_estimators, max_features):

        params = {
            'max_samples': max_samples,
            'max_features':max_features,
            'n_estimators':int(n_estimators)
        }
        clf = RandomForestClassifier(max_samples=params['max_samples'], max_features=params['max_features'], n_estimators=params['n_estimators'])
        score = self.stratified_kfold_score(clf)
        return score

    def bo_params(self):
        rf_bo = BayesianOptimization(self.bo_params_rf, {
                                                    'max_samples':(0.5,1),
                                                    'max_features':(0.5,1),
                                                    'n_estimators':(100,200)
                                                    }, random_state=0)
        results = rf_bo.maximize(n_iter=70, init_points=20, acq='ei')
        params = rf_bo.max['params']
        params['n_estimators']= int(params['n_estimators'])
        print(params)
        pd.DataFrame(params, orient='index', columns=['param', 'value']).to_csv('src\\modelling\\rf_params.csv')

        return params

    
    def bo_results(self):
        target_names = ['Reliable', 'Unreliable']
        params = self.bo_params()

        rf_v1 = RandomForestClassifier(max_samples=params['max_samples'],max_features=params['max_features'],n_estimators=params['n_estimators'])
        rf_v1.fit(self.train, self.y_train)
        y_pred = rf_v1.predict(self.test)

        report = classification_report(self.y_test, y_pred, target_names=target_names, output_dict=True)
        print(report)
        final = pd.DataFrame(report).transpose()

        final.to_csv('src\\modelling\\final_bo_results.csv', index=False)
            




    # if __name__ == "__main__":
        