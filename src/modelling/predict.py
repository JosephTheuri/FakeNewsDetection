import joblib
from sklearn.metrics import classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


def predict_model(model, df):
    columns = pd.read_csv("D:\JT\GTU OMS\Data and Visual Analytics (CSE 6242)\Project\src\processing\columns.csv")
    columns = list(columns['columns'])
    labels =df['label']
    clf = joblib.load("model\{}.pkl".format(model))
    print('Model loaded successfully')

    # Scale data
    scaler = StandardScaler()
    df_s = scaler.fit_transform(df[columns])
    print('Scaling complete')

    y_pred = clf.predict(df_s)
    y_pred = [int(float(x)) for x in y_pred]
    report = classification_report(labels, y_pred, output_dict=True)
    print(classification_report(labels, y_pred))
    print("balanced accuracy: ", balanced_accuracy_score(labels, y_pred))
    final = pd.DataFrame(report).transpose()

    final.to_csv('src\\modelling\\predicted_results.csv', index=False)


# if __name__=="__main__":
