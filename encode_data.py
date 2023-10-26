import pandas as pd
from sklearn.preprocessing import LabelEncoder

test_csv = pd.read_csv("test.csv")
train_csv = pd.read_csv("train.csv")

print(test_csv.head(5))
print(train_csv.head(5))

#def encode_function():
print(test_csv.MSSubClass.dtype)


def encode_function(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col]) 

    return df
        
test_encoded = encode_function(test_csv)
train_encoded = encode_function(train_csv)

print(test_encoded.head(5))
print(train_encoded.head(5))

print(train_encoded.dtypes)

test_encoded.to_csv('test_encoded.csv')
train_encoded.to_csv('train_encoded.csv')
