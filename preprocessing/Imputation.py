import pandas as pd
from sklearn.impute import SimpleImputer

TH = "MISSING VALUE THRESHOLD"

## Read your genetic marker data 
## n x m: n is the total number of lines and m is the total number of markers
data = pd.read_csv("YOUR GEMETIC MARKER DATA")

## Remove markers that have more missing values than the threshold
frequency_missing = data[data=='-'].count() / data.shape[0]
data = data.loc[:,list(frequency_missing[frequency_missing<TH].index)]

## Impute missing values with the most frequent allele in each marker
imp = SimpleImputer(missing_values='-', strategy='most_frequent')
imp.fit(data)
data_filtered = pd.DataFrame(data=imp.transform(data), columns=data.columns)

## Save the imputated data
data_filtered.to_csv("IMPUTATED GENETIC MARKER DATA")