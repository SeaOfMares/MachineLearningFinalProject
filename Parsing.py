import pandas as pd
import numpy as np
import csv
from Encoding import Encoder
df = pd.read_csv("Updated_Accidents_Without_Negatives.csv")
df = df.dropna()
org_data = df.to_numpy()
encoder = Encoder()
headers = []
for i in range(80):
    temp_str = "h" + str(i)
    headers.append(temp_str)
#Printing the data to csv
with open('Transformed_Accidents.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        #write the header
        writer.writerow(headers)
        #print(df.isnull().sum())
        new_data = np.array([])
        for i in range(org_data.shape[0]):
                row = org_data[i]
                new_row = encoder.encodeRow(row)
                writer.writerow(new_row)


