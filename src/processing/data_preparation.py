from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class Preprocess():
    def __init__(self, data, language):
        self.data = data
        self.language = language

    def data_split(self):
        labels = self.data['label']
        text = self.data['text']
        #DataFlair - Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=123, stratify=labels)
        return x_train, x_test, y_train, y_test

    def vectorize_data(self):
        x_train, x_test, y_train, y_test = self.data_split()
        print("\nDataset Split:", 
            '\nX_train:', x_train.shape[0], 
            '\ny_train:', y_train.shape[0], 
            '\nX_test:', x_test.shape[0], 
            '\ny_test:', y_test.shape[0], '\n' , '-----'*10
        )

        #DataFlair - Initialize a TfidfVectorizer
        tfidf_vectorizer=TfidfVectorizer(stop_words=self.language, max_df=0.7, lowercase= True)

        #DataFlair - Fit and transform train set, transform test set
        tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
        tfidf_test = tfidf_vectorizer.transform(x_test)
        return tfidf_train, tfidf_test, y_train, y_test

