import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

class CascadingClassifier:
    def __init__(self):
        # Load all models in cascade
        self.lr_model = load_model('models/logistic_model.h5')
        self.rf_model = joblib.load('models/rf_model.joblib')
        self.cnn_model = load_model('models/cnn_model.h5')
        
        # Confidence thresholds for each stage
        self.lr_threshold = 0.95  # If logistic regression is 95% confident
        self.rf_threshold = 0.85  # If random forest is 85% confident
    
    def predict(self, x):
        # Reshape if single sample
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # First stage: Logistic Regression
        lr_probs = self.lr_model.predict(x)
        lr_pred = np.argmax(lr_probs, axis=1)
        lr_max_prob = np.max(lr_probs, axis=1)
        
        # Initialize final predictions
        final_pred = np.zeros(x.shape[0], dtype=int)
        
        # Get indices of samples where LR is confident
        confident_mask = lr_max_prob >= self.lr_threshold
        final_pred[confident_mask] = lr_pred[confident_mask]
        
        # Remaining samples go to next stage
        remaining_indices = np.where(~confident_mask)[0]
        if len(remaining_indices) > 0:
            x_remaining = x[remaining_indices]
            
            # Second stage: Random Forest
            rf_probs = self.rf_model.predict_proba(x_remaining)
            rf_pred = np.argmax(rf_probs, axis=1)
            rf_max_prob = np.max(rf_probs, axis=1)
            
            # Get indices where RF is confident
            rf_confident_mask = rf_max_prob >= self.rf_threshold
            final_rf_confident = remaining_indices[rf_confident_mask]
            final_pred[final_rf_confident] = rf_pred[rf_confident_mask]
            
            # Remaining samples go to final stage
            final_remaining = remaining_indices[~rf_confident_mask]
            if len(final_remaining) > 0:
                x_final = x[final_remaining]
                if len(x_final.shape) == 2:  # Need to reshape for CNN
                    x_final = x_final.reshape((-1, 28, 28, 1))
                
                # Final stage: CNN
                cnn_pred = np.argmax(self.cnn_model.predict(x_final), axis=1)
                final_pred[final_remaining] = cnn_pred
        
        return final_pred
    
    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy

if __name__ == '__main__':
    # Example usage
    import tensorflow as tf
    
    # Load data
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape((-1, 28*28)).astype('float32') / 255.0
    
    # Initialize and evaluate cascading classifier
    classifier = CascadingClassifier()
    accuracy = classifier.evaluate(x_test[:1000], y_test[:1000])  # Test on subset
    
    print(f"Cascading Classifier Accuracy: {accuracy:.4f}")
