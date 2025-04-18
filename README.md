```markdown
# Cascading Machine Learning Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.8-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-1.0.2-green?style=for-the-badge&logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements a cascading classifier system for image recognition that:
1. First uses a simple logistic regression model for easy-to-classify images
2. Then applies a random forest classifier for moderately difficult cases
3. Finally uses a convolutional neural network (CNN) for the most challenging images

The cascade approach provides an optimal balance between computational efficiency and classification accuracy.

## Features

- **Three-stage cascading architecture**
- **Dynamic routing** of samples based on confidence thresholds
- **Pre-trained models** included for MNIST digit classification
- **Easy extensibility** for other classification tasks
- **Comprehensive evaluation** of cascade performance

## Installation

1. Clone the repository:
```bash
git clone https:/[(https://github.com/balaji7256/ML_PROJECT_CASCADING.git)]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models

To train all models in the cascade:
```bash
python train_models.py
```

This will:
- Download MNIST dataset
- Train logistic regression, random forest, and CNN models
- Save models to the `models/` directory

### 2. Run Cascading Classifier

To evaluate the cascading classifier on test data:
```bash
python cascading_classifier.py
```

### 3. Custom Usage

To use the cascading classifier in your own code:
```python
from cascading_classifier import CascadingClassifier

# Initialize classifier
classifier = CascadingClassifier()

# Make predictions
predictions = classifier.predict(your_data)

# Evaluate accuracy
accuracy = classifier.evaluate(x_test, y_test)
```

## Project Structure

```
cascading-ml-classifier/
├── data/                   # Sample data (not included, downloaded automatically)
├── models/                 # Saved model files
│   ├── logistic_model.h5   # Logistic regression model
│   ├── rf_model.joblib     # Random forest model
│   └── cnn_model.h5       # CNN model
├── train_models.py         # Script to train all models
├── cascading_classifier.py # Main cascading classifier implementation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Results

On the MNIST test set (10,000 samples), the cascading classifier achieves:

| Metric            | Value |
|-------------------|-------|
| Overall Accuracy  | 98.3% |
| Stage 1 Usage     | 68%   |
| Stage 2 Usage     | 25%   |
| Stage 3 Usage     | 7%    |
| Speed Improvement | 3.2x  |

Compared to using just the CNN:
- 3.2x faster inference
- Only 0.4% accuracy drop
- 93% of samples processed by simpler models

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/yourfeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/yourfeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
