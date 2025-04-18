import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import os

# Load and preprocess data (MNIST as example)
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28*28)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28*28)).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

# Train simple logistic regression model (first in cascade)
def train_logistic_model(x_train, y_train):
    model = tf.keras.Sequential([
        layers.Dense(10, activation='softmax', input_shape=(784,))
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64)
    return model

# Train random forest (second in cascade)
def train_rf_model(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    return model

# Train CNN (final in cascade)
def train_cnn_model(x_train, y_train):
    x_train = x_train.reshape((-1, 28, 28, 1))
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64)
    return model

def main():
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Train models
    print("Training Logistic Regression...")
    lr_model = train_logistic_model(x_train, y_train)
    lr_model.save('models/logistic_model.h5')
    
    print("\nTraining Random Forest...")
    rf_model = train_rf_model(x_train, y_train)
    joblib.dump(rf_model, 'models/rf_model.joblib')
    
    print("\nTraining CNN...")
    cnn_model = train_cnn_model(x_train, y_train)
    cnn_model.save('models/cnn_model.h5')
    
    print("\nAll models trained and saved!")

if __name__ == '__main__':
    main()
