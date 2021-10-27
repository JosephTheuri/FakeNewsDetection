import pandas as pd

class ImportData():
    def __init__(self, path):
        self.path = path

    # Read csv data
    def read_csv(self):
        df = pd.read_csv(self.path)
        return df

    # Clean dataframe
    def clean_data(self):
        df = self.read_csv()

        # Remove rows with missing text values
        df_no_na = df[df['text'].notna()]
        print('Null rows removed:', len(df)-len(df_no_na))
        
        # Drop duplicates
        df_no_duplicates = df_no_na.drop_duplicates(subset='Text')
        print('Duplicates removed:', len(df)-len(df_no_duplicates))

        return df_no_duplicates

    # Print pandas properties
    def print_pstats(self):
        df = self.clean_data()
        print("\nThe dataset has the following properties:", "\nRows:", df.shape[0], "\nColumns:", df.shape[1], "|| Names -", list(df.columns)) 

        # Label Properies
        print('\nBelow is the distribution of the target class')
        print(df['label'].value_counts(normalize=True)*100 , '\n' , '-----'*10)
        return df

if __name__ == "__main__":
    ImportData(path="data\\raw\\mock_train.csv").print_pstats()
