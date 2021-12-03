import csv
import pandas as pd

class ImportData():
    def __init__(self, path):
        self.path = path

    # Read csv data
    def read_csv(self):
        df = pd.read_csv(self.path, error_bad_lines=False, engine='python')
        df = df.rename(columns={'content':'text', 'Body':'text', 'Fake':'label'})[['text', 'label']]
        return df

    # Clean dataframe
    def clean_data(self):
        df = self.read_csv()

        # Remove rows with missing text values
        df_no_na = df[df['text'].notna()]
        print('Null rows removed:', len(df)-len(df_no_na))
        
        # Drop duplicates
        df_no_duplicates = df_no_na.drop_duplicates(subset='text')
        print('Duplicates removed:', len(df_no_na)-len(df_no_duplicates))

        # Only relevant target varibales
        df['label'] = df['label'].astype(str)
        valid_labels = df['label'].value_counts(normalize=True)[df['label'].value_counts(normalize=True)>0.05].index[:1]
        df_valid_labels = df_no_duplicates[df['label'].isin(valid_labels)]
        # df_valid_labels = df_no_duplicates[df_no_duplicates['label'].isin(['0','1'])]
        print('Bad Labels removed:', len(df_no_duplicates)-len(df_valid_labels))

        return df_valid_labels

    # Print pandas properties
    def print_pstats(self):
        df = self.clean_data()
        print("\nThe dataset has the following properties:", "\nRows:", df.shape[0], "\nColumns:", df.shape[1], "|| Names -", list(df.columns)) 

        # Label Properies
        print('\nBelow is the distribution of the target class')
        print(df['label'].value_counts(normalize=True)*100 , '\n' , '-----'*10)
        return df

if __name__ == "__main__":
    ImportData(path="data\\raw\\translated_dataset_all.csv").print_pstats()
    # data\raw\translated_test_5k_english.csv
    # data\\raw\\NELA_20_subset_200k.csv

