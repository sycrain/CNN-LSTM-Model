import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical, plot_model
from keras.callbacks import CSVLogger

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

X = X.reshape(X.shape[0], 500, 2)  
Y_onehot = to_categorical(Y, num_classes=6)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

def build_cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(500, 2)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))  
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

cnn_model = build_cnn_model()
cnn_model.summary()
plot_model(cnn_model, to_file='cnn_model_structure.png', show_shapes=True)  

csv_logger = CSVLogger('cnn_training_log.csv', append=False, separator=';')
history = cnn_model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=32, callbacks=[csv_logger])

with open('cnn_training_results.txt', 'w') as f:
    for i in range(len(history.history['loss'])):
        f.write(f"Epoch {i+1}, Loss: {history.history['loss'][i]}, Accuracy: {history.history['accuracy'][i]}, "
                f"Val Loss: {history.history['val_loss'][i]}, Val Accuracy: {history.history['val_accuracy'][i]}\n")

Y_pred = cnn_model.predict(X_test)
Y_pred_labels = np.argmax(Y_pred, axis=1)
Y_true_labels = np.argmax(Y_test, axis=1)

accuracy = accuracy_score(Y_true_labels, Y_pred_labels)
precision = precision_score(Y_true_labels, Y_pred_labels, average='weighted')
recall = recall_score(Y_true_labels, Y_pred_labels, average='weighted')
f1 = f1_score(Y_true_labels, Y_pred_labels, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

conf_mat = confusion_matrix(Y_true_labels, Y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil'],
            yticklabels=['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig('confusion_matrix.png')
plt.show()

cnn_model.save('cnn_model_weights.h5')

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.show()
