import pandas as pd

def load():
    dataset_path = '../data/heart.csv'
    df = pd.read_csv(dataset_path)  # adjust path as needed
    
    print("====================================================================================================")
    print("Number of rows: ")
    print(df.shape[0])
    
    print("====================================================================================================")
    print("Columns: ")
    print(df.columns)
    
    print("====================================================================================================")
    print("Number of columns: ")
    print(len(df.columns))
    
    print("====================================================================================================")
    print("Missing values per column:")
    print(df.isnull().sum())
    
    print("====================================================================================================")
    print("Cleaned dataset: ")
    df = df.dropna()
    print(df.head(5))
    
    print("====================================================================================================")
    
    X = df.drop('target', axis=1)  # feature set
    y = df['target']               # target set
    
    return X, y

# Call the function
X, y = load()
