import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import csv
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTENC

df = pd.read_csv("Transformed_Accidents.csv")
arr = df.to_numpy(dtype= int)
labels = arr.transpose()[0] #getting the classes
features = arr.transpose()[1:] #getting the features and excluding the classes
features = features.transpose()

f = open("Ensemble_Models_Results.csv", 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
writer.writerow(["Splitter", "Criterion", "Max_Depth", "Accuracy"])

smote_nc = SMOTENC(categorical_features= [1,2,3], random_state=0)
x_resampled , y_resampled = smote_nc.fit_resample(features,labels)

x_train , x_test, y_train, y_test = train_test_split(
    x_resampled, y_resampled, test_size= 0.2, random_state=42
)

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes = np.unique(y_train), y =y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

best_accuracy = 0
best_depth = 20
splitter_a = "random"
criterion_a = ["entropy","gini","log_loss"]
best_criterion = ""
for criter in criterion_a:
    row = [splitter_a,criter,0,0]
    print("------------------------")
    print("Criterion : " , criter)
    for i in range(50,60):
        model = RandomForestClassifier(criterion= criter, max_depth= i)
        model.fit(X=x_train,y=y_train)
        predictions = model.predict(X=x_test)
        temp_accuracy = accuracy_score(y_test,predictions)
        print("Accuracy for max depth at ",i,": ",temp_accuracy)

        if temp_accuracy > best_accuracy:
            best_depth = i
            best_accuracy = temp_accuracy
            best_model = model
            best_criterion = criter
            #best_predictions = predictions

        row[2] = i
        row[3] = temp_accuracy
        writer.writerow(row)


print("---------------------------")
print("Best Max Depth is", best_depth)
print("Best Criterion:", best_criterion)
print(f'Accuracy {best_accuracy:.4}')