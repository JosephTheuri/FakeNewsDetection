import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import feature_selection
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from processing.data_cleaning import DataCleaning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocess():
    def __init__(self, data, language):
        self.data = data.dropna()
        self.language = language

    def feature_selection(self):
        df = self.data[[x for x in self.data.columns if '_' in x]]
        labels = self.data['label'].astype(str)

         # Drop highly correlated columsn
        corr_matrix = df.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.9)]
        df1 = df.drop(df.columns[to_drop], axis=1)
        print('Corr Matrix removed {} columns'.format(df.shape[1]-df1.shape[1]))

        
        # Drop columns with low variance
        selector = VarianceThreshold()
        selector.fit(df1)
        df2 = df1[df1.columns[selector.get_support(indices=True)]]
        print('VarianceThreshold removed {} columns'.format(df1.shape[1]-df2.shape[1]))

        # Recursive feature elimination with cross validation and random forest classification
        # clf_rf_4 = RandomForestClassifier() 
        # rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='roc_auc')   #5-fold cross-validation

        # # split data train 70 % and test 30 %
        # x_train, x_test, y_train, y_test = train_test_split(df2, labels, test_size=0.1, random_state=123)
        # rfecv = rfecv.fit(x_train, y_train)

        # rf_columns = [x for x in x_train.columns[rfecv.support_]]
        # print('Optimal number of features :', rfecv.n_features_)
        # print('Best features :', rf_columns)

        # df3 = df2[rf_columns].dropna()
        # print('RF removed {} columns'.format(df2.shape[1]-df3.shape[1]))
        pd.DataFrame({'columns': df2.columns}).to_csv("src\processing\columns.csv")

        return df2.dropna()

    def data_split(self):
        # Split labels and training data
        df = self.feature_selection()
        # df = self.data[[x for x in self.data.columns if '_' in x]]
        columns = df.columns
        labels = self.data['label'].astype(str)
        labels = labels.drop([x for x in labels.index if x not in df.index])
        

        # Scale data
        scaler = StandardScaler()
        df_s = scaler.fit_transform(df)


        #DataFlair - Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(df_s, labels, test_size=0.2, random_state=123, stratify=labels)

        # Perform smoote rebalancing
        if abs(labels.value_counts(normalize=True)[0] - labels.value_counts(normalize=True)[1]) > 0.05:
            sm = SMOTE(random_state=0)
            x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())
            print('SMOTE Performed')
        else:
            x_train_res, y_train_res = x_train, y_train
            print('SMOTE skipped')
        
        print("\nDataset Split:", 
            '\nX_train:', x_train_res.shape[0], 
            '\ny_train:', y_train_res.shape[0], 
            '\nX_test:', x_test.shape[0], 
            '\ny_test:', y_test.shape[0], '\n' , '-----'*10
        )


        return x_train_res, x_test, y_train_res, y_test, columns

    def vectorize_data(self):
        df = self.data
        text = DataCleaning(data = df['text']).clean_data()
        columns = df.columns
        labels = self.data['label'].astype('str')

        #DataFlair - Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=0, stratify=labels)


        print("\nDataset Split:", 
            '\nX_train:', x_train.shape[0], 
            '\ny_train:', y_train.shape[0], 
            '\nX_test:', x_test.shape[0], 
            '\ny_test:', y_test.shape[0], '\n' , '-----'*10
        )

        #DataFlair - Initialize a TfidfVectorizer
        tfidf_vectorizer=TfidfVectorizer(stop_words=self.language, lowercase= True, ngram_range=(1,3),
                                        strip_accents='unicode', analyzer='word',
                                        use_idf=1,smooth_idf=1,sublinear_tf=1,)

        #DataFlair - Fit and transform train set, transform test set
        tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
        tfidf_test = tfidf_vectorizer.transform(x_test)
        return tfidf_train, tfidf_test, y_train, y_test, columns

