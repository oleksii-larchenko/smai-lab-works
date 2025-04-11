import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if len(data) != 15:
            continue
        label = data[-1]
        features = data[:-1]
        if label == '<=50K' and count_class1 < max_datapoints:
            X.append(features)
            y.append(label)
            count_class1 += 1
        elif label == '>50K' and count_class2 < max_datapoints:
            X.append(features)
            y.append(label)
            count_class2 += 1

X = np.array(X)
y = np.array(y)

label_encoders = []
X_encoded = np.empty(X.shape)

for i in range(X.shape[1]):
    try:
        X_encoded[:, i] = X[:, i].astype(float)
        label_encoders.append(None)  # Числовий стовпець
    except ValueError:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoders.append(encoder)

scaler = StandardScaler()
X_encoded = scaler.fit_transform(X_encoded)

label_encoder_y = preprocessing.LabelEncoder()
y = label_encoder_y.fit_transform(y)

print("Унікальні класи y:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=5)

print("Мітки y_train:", np.unique(y_train, return_counts=True))

classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\n=== Метрики якості ===")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print(f"Precision: {round(precision_score(y_test, y_pred, average='weighted') * 100, 2)}%")
print(f"Recall: {round(recall_score(y_test, y_pred, average='weighted') * 100, 2)}%")
print(f"F1 Score: {round(f1_score(y_test, y_pred, average='weighted') * 100, 2)}%")

f1_cv = cross_val_score(classifier, X_encoded, y, scoring='f1_weighted', cv=3)
print(f"F1 (cross-val): {round(f1_cv.mean() * 100, 2)}%")

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male',
              '0', '0', '40', 'United-States']

input_data_encoded = []
for i, item in enumerate(input_data):
    if label_encoders[i] is None:
        input_data_encoded.append(float(item))
    else:
        input_data_encoded.append(label_encoders[i].transform([item])[0])

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
input_data_encoded = scaler.transform(input_data_encoded)

predicted_class = classifier.predict(input_data_encoded)

result = label_encoder_y.inverse_transform(predicted_class)[0]

print("\n=== Результат для тестової точки ===")
print(f"Прогнозований клас: {result}")
