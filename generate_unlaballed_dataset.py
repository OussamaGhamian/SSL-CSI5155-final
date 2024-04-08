import pandas as pd
from sklearn.model_selection import train_test_split

#Split the dataset into labell and unlablled data
#20% of the data will be labelled and the 80% unlaballed

data = pd.read_csv('./bankDataset/bank-full.csv', sep=';')

labeled_data, unlabeled_data = train_test_split(data, test_size=0.8, random_state=42)

# Drop labels from the unlabeled subset
unlabeled_data = unlabeled_data.drop('y', axis=1)

df = pd.concat([labeled_data,unlabeled_data],ignore_index=True)

df.to_csv("./bankDataset/label_unlabel.csv",index=False)