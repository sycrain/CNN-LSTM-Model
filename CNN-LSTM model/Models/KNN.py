import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data_from_folders(folder_paths):
    X = []
    Y = []
    for label, folder in enumerate(folder_paths):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file_path.endswith(".txt"):
                data = np.loadtxt(file_path)  
                X.append(data.flatten())  
                Y.append(label)
    return np.array(X), np.array(Y)
folder_paths = ['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil']
X, Y = load_data_from_folders(folder_paths)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
history = {'accuracy': []}
for epoch in range(1, 101):  
    knn_model.fit(X_train, Y_train)
    train_acc = knn_model.score(X_train, Y_train)
    test_acc = knn_model.score(X_test, Y_test)

    print(f'Epoch {epoch}: Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    history['accuracy'].append((train_acc, test_acc))

with open('knn_training_results.txt', 'w') as f:
    f.write('Epoch\tTrain Accuracy\tTest Accuracy\n')
    for epoch in range(100):
        f.write(f"{epoch+1}\t{history['accuracy'][epoch][0]:.4f}\t{history['accuracy'][epoch][1]:.4f}\n")
Y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
conf_mat = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil'],
            yticklabels=['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()