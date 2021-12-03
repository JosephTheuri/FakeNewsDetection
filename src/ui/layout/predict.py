import joblib
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys

from pandas.core.algorithms import mode 
path = "D:\JT\GTU OMS\Data and Visual Analytics (CSE 6242)\Project\src\processing"
sys.path.insert(1, path)
from feature_extraction import FeatureExtraction

def get_prediction(text):
    columns = pd.read_csv("D:\JT\GTU OMS\Data and Visual Analytics (CSE 6242)\Project\src\processing\columns.csv")['columns']
    data = FeatureExtraction(pd.DataFrame({'text':[text]}), language='english').extract_features()[columns]

    # Create word cload
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    word_cloud.to_file('D:\JT\GTU OMS\Data and Visual Analytics (CSE 6242)\Project\src\processing\word-cloud.jpg')

    #get all files for translation
    modelDirectory = pathlib.Path("D:\JT\GTU OMS\Data and Visual Analytics (CSE 6242)\Project\model")
    model_results = {'model':[], 'prediction':[], '0.0':[], '1.0':[]}
    for model in modelDirectory.iterdir():
        # Save model names
        model_name = str(model).split('.')[0].split('\\')[-1]

        # Load model
        clf = joblib.load(model)
        try:
            proba = clf.predict_proba(data)[0]
        except:
            proba = [None, None]
        model_results['model'].append(model_name)
        model_results['prediction'].append(clf.predict(data)[0])
        model_results['0.0'].append(proba[0])
        model_results['1.0'].append(proba[1])
        if model_name == "RandomForest":
            feat_importances =  pd.Series(clf.feature_importances_,
                                            index= columns
                                            ).sort_values(ascending=False)
    
    feat_importances = ['{} - {}%'.format(str(x).replace('_',' ').title(),int(y*100)) for x, y in zip(feat_importances.index, list(feat_importances))]
    model_results = pd.DataFrame(model_results)
    model_data = {'Model':list(model_results.model)} 
    values = []
    for x in range(model_results.shape[0]):
        try:
            u = int(model_results['1.0'][x]*100)
            r = int(model_results['0.0'][x]*100)
            values.append('Prediction: Reliable with probability: {}% and Prediction: Unreliable with probability: {}%'.format(r,u))
        except:
            p =  'Reliable' if model_results['prediction'][x] == '0.0' else 'Unreliable'
            values.append('Prediction: {}'.format(p))
    model_data['Results'] = values
    model_data = pd.DataFrame(model_data).to_dict('records')
    label = model_results['prediction'].mode()
    prob = '{}/10'.format(int(round(model_results[label].mean()*10)))


    return  {'label':label, 'score':prob, 'fi':feat_importances, 'model_data':model_data}


test = '''
The initial size of the control. This value is in pixels unless the value of the type attribute is text or password, 
in which case it is an integer number of characters. Starting in, this attribute applies only when the type attribute 
is set to text, search, tel, url, email, or password, otherwise it is ignored. In addition, the size must be greater than zero. 
If you do not specify a size, a default value of 20 is used.' simply states "the user agent should ensure that at least that 
many characters are visible", but different characters can have different widths in certain fonts. In some browsers, a certain 
string with x characters will not be entirely visible even if size is defined to at least x.
'''

if __name__ == '__main__':
    x = get_prediction(test)
    print(x)
    