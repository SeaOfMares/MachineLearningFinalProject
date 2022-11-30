import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTENC

df = pd.read_csv("Transformed_Accidents.csv")
arr = df.to_numpy(dtype= int)
labels = arr.transpose()[0]
features = arr.transpose()[1:]
features = features.transpose()

smote_nc = SMOTENC(categorical_features= [1,2,3], random_state=0)
x_resampled , y_resampled = smote_nc.fit_resample(features,labels)


kf = KFold(n_splits= 10)
for train_index, test_index in kf.split(x_resampled):
    x_train,x_test = x_resampled[train_index], x_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes = np.unique(y_train), y =y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))

f = open("Decision_Tree_Models_Result.csv", 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
writer.writerow(["Splitter", "Criterion", "Max_Depth", "Accuracy", "Precision", "Recall", "F1_Score"])

best_accuracy = 0
best_depth = 5
best_recall= 0
best_precision = 0
best_f1 = 0
splitter_a = ["best" ,"random"]
criterion_a = ["entropy","gini","log_loss"]
row = ["","",0,0,0,0,0]

for split in splitter_a:
    row[0] = split
    for crit in criterion_a:
        row[1] = crit
        print("---------------------------")
        print("Splitter:",split, " | Criterion:", crit)
        for i in range(20,61):
            modelTree = DecisionTreeClassifier(criterion= crit, splitter = split, max_depth= i, class_weight = class_weights)
            sampler = RandomOverSampler(random_state=0)
            model = make_pipeline(sampler, modelTree)
            model.fit(X=x_train,y=y_train)
            predictions = model.predict(X=x_test)

            temp_accuracy = accuracy_score(y_test,predictions)
            precision = precision_score(y_test, predictions, average="weighted")
            recall = recall_score(y_test,predictions,average="weighted")
            f1 = f1_score(y_test,predictions, average= "weighted")
            print("Depth: ",i,f'| Accuracy: {temp_accuracy:.4}', f'| Precision: {precision:.4}', f'| Recall: {recall:.4}', f'| F1: {f1:.4}')

            if temp_accuracy > best_accuracy:
                best_accuracy_depth = i
                best_accuracy = temp_accuracy
                best_accuracy_model = model
                best_accuracy_criterion = crit
                best_accuracy_splitter = split

            if recall > best_recall:
                best_recall_depth = i
                best_recall = recall
                best_recall_criterion = crit
                best_recall_splitter = split

            if precision > best_precision:
                best_precision_depth = i
                best_precision = precision
                best_precision_criterion = crit
                best_precision_splitter = split

            if f1 > best_f1:
                best_f1_depth = i
                best_f1 = f1
                best_f1_criterion = crit
                best_f1_splitter = split

            row[2] = i
            row[3] = temp_accuracy
            row[4] = precision
            row[5] = recall
            row[6] = f1
            writer.writerow(row)


print("---------------------------")
print("Accuracy")
print("Best Max Depth is", best_accuracy_depth)
print("Best Criterion:", best_accuracy_criterion)
print("Best Splitter:", best_accuracy_splitter)
print(f'Accuracy {best_accuracy:.4}')
print("---------------------------")
print("Precision")
print("Best Max Depth is", best_precision_depth)
print("Best Criterion:", best_precision_criterion)
print("Best Splitter:", best_precision_splitter)
print(f'Accuracy {best_precision:.4}')
print("---------------------------")
print("Recall")
print("Best Max Depth is", best_recall_depth)
print("Best Criterion:", best_recall_criterion)
print("Best Splitter:", best_recall_splitter)
print(f'Accuracy {best_recall:.4}')
print("---------------------------")
print("F1 Score")
print("Best Max Depth is", best_f1_depth)
print("Best Criterion:", best_f1_criterion)
print("Best Splitter:", best_f1_splitter)
print(f'Accuracy {best_f1:.4}')
print("---------------------------")
print("Bins Predictions :", np.bincount(predictions))
print("Bins y_test :", np.bincount(y_test))