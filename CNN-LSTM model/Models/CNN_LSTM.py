from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from keras.callbacks import CSVLogger, Callback
import matplotlib.pyplot as plt
import itertools

# Load data from folders
def load_data_from_folders(folder_paths):
    X_data = []
    Y_labels = []
    for label, folder in enumerate(folder_paths):
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder, filename)
                data = np.loadtxt(file_path)
                X_data.append(data[:, 1])  # Use the second column as features
                Y_labels.append(label)  # Label is folder index
    X_data = np.array(X_data)
    Y_labels = np.array(Y_labels)
    return X_data, Y_labels

# Define folder paths
folder_paths = ['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil']

# Load data
X, Y = load_data_from_folders(folder_paths)
X = X.reshape(-1, 500, 1)  # Reshape to (samples, 500, 1)

# One-hot encode labels
Y_onehot = np_utils.to_categorical(Y)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=0)

# Create CNN-LSTM model
def create_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, strides=2, input_shape=(500, 1), padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5, strides=2, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Custom callback to save loss and accuracy for each epoch
class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

# Save training logs
csv_logger = CSVLogger('training_log_r.csv', append=False)
loss_history = LossHistory()

# Create model
model = create_model()

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test),
                    callbacks=[csv_logger, loss_history])

# Save losses and accuracy for each epoch
with open('loss_and_accuracy_r.txt', 'w') as f:
    for i in range(len(loss_history.losses)):
        f.write(f"{i+1}, {loss_history.losses[i]}, {loss_history.accuracy[i]}\n")

# Predict on the test set
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Calculate overall classification metrics
overall_accuracy = accuracy_score(Y_true, Y_pred_classes)
overall_precision = precision_score(Y_true, Y_pred_classes, average='weighted')
overall_recall = recall_score(Y_true, Y_pred_classes, average='weighted')
overall_f1 = f1_score(Y_true, Y_pred_classes, average='weighted')

print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1}")

# Calculate metrics for each class
class_accuracy = {}
class_precision = precision_score(Y_true, Y_pred_classes, average=None)
class_recall = recall_score(Y_true, Y_pred_classes, average=None)
class_f1 = f1_score(Y_true, Y_pred_classes, average=None)

# Print metrics for each class
target_names = ['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil']
for i, target_name in enumerate(target_names):
    class_accuracy[target_name] = accuracy_score(Y_true[Y_true == i], Y_pred_classes[Y_true == i]) if sum(Y_true == i) > 0 else 0
    print(f"{target_name} - Accuracy: {class_accuracy[target_name]:.2f}, Precision: {class_precision[i]:.2f}, Recall: {class_recall[i]:.2f}, F1 Score: {class_f1[i]:.2f}")

# Confusion matrix
cm = confusion_matrix(Y_true, Y_pred_classes)

# Plot and save confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix_r1.png', dpi=1200, bbox_inches='tight')
    plt.show()

# Plot and save confusion matrix
plot_confusion_matrix(cm, classes=target_names)
