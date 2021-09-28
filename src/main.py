from preparation.import_data import ImportData
from processing.data_preparation import Preprocess
from modelling.train_model import TrainModel


if __name__ == "__main__":
    data = ImportData(path="data\\raw\\mock_train.csv").print_pstats()
    tfidf_train, tfidf_test, y_train, y_test = Preprocess(data=data, language='english').vectorize_data()
    TrainModel(tfidf_train, tfidf_test, y_train, y_test).train_results()
