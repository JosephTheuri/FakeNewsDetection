import pandas as pd
from preparation.import_data import ImportData
from processing.data_preparation import Preprocess
from processing.feature_extraction import FeatureExtraction
from modelling.train_model import TrainModel
from modelling.optimization import OptimizeRF
from modelling.predict import predict_model


if __name__ == "__main__":
    path = "data\\raw\\translated_dataset_all.csv"
    data = ImportData(path=path).print_pstats()
    data_plus = FeatureExtraction(data, language='english').extract_features()
    data_plus.to_csv('src\\processing\\data_features.csv')
    data_plus = pd.read_csv('src\\processing\\data_features.csv')

    X_train, X_test, y_train, y_test, columns = Preprocess(data=data_plus, language='english').data_split()
    top = TrainModel(X_train, X_test, y_train, y_test, columns).train_results()

    # model = 'SVC'
    model = top
    path = "data\\raw\German_News_Dataset.csv"

    data = ImportData(path=path).print_pstats()
    data_plus = FeatureExtraction(data, language='german').extract_features()
    data_plus.to_csv('src\\processing\\data_features_german.csv')
    data_plus = pd.read_csv('src\\processing\\data_features_german.csv')
    predict_model(model, data_plus)
