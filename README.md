# ML Models From Scratch

A comprehensive educational resource for understanding machine learning algorithms from the ground up. This library provides detailed mathematical foundations, pure NumPy implementations, and practical examples using publicly available datasets.

## 🎯 Project Goal

Create a complete learning resource for students and practitioners of machine learning by:
- Deriving the mathematics behind each algorithm step-by-step
- Implementing models from scratch using only NumPy (no scikit-learn/TensorFlow/PyTorch)
- Providing real-world examples with publicly available datasets
- Explaining every line of code and mathematical concept

## 📚 What's Included

### Current Models

#### Linear Models
- **Linear Regression** - Gradient descent with MSE loss
- **Ridge Regression** - Linear regression with L2 regularization
- **Logistic Regression** - Binary classification with cross-entropy loss

#### Coming Soon
- **Lasso Regression** - L1 regularization for feature selection
- **Elastic Net** - Combined L1 and L2 regularization
- **Polynomial Regression** - Non-linear relationships
- **Decision Trees** - Tree-based learning
- **Random Forests** - Ensemble methods
- **Support Vector Machines (SVM)** - Margin-based classification
- **K-Nearest Neighbors (KNN)** - Instance-based learning
- **Naive Bayes** - Probabilistic classification
- **K-Means Clustering** - Unsupervised learning
- **Principal Component Analysis (PCA)** - Dimensionality reduction
- **Neural Networks** - Multi-layer perceptrons from scratch
- **Advanced NN** - CNNs, RNNs

## 📖 Prerequisites

### Mathematics Knowledge Required

#### Linear Algebra (Essential)
- Vectors and matrices
- Matrix multiplication and transpose
- Dot products
- Vector norms (L1, L2)
- Basic understanding of vectorization

#### Calculus (Essential)
- Partial derivatives
- Chain rule
- Gradient descent concept
- Understanding of optimization

#### Statistics & Probability (Recommended)
- Mean, variance, standard deviation
- Probability distributions (for logistic regression, naive bayes)
- Maximum likelihood estimation
- Basic hypothesis testing

#### Resources to Learn Prerequisites
- **Linear Algebra**: [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- **Calculus**: [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- **Statistics**: Khan Academy Statistics and Probability

### Python Knowledge Required

#### Core Python (Essential)
- Variables, data types, and operators
- Control flow (if/else, loops)
- Functions and basic OOP (classes, methods)
- List comprehensions and basic data structures

#### NumPy (Essential)
- Array creation and indexing
- Array operations and broadcasting
- Basic linear algebra operations (`np.dot`, `np.sum`, etc.)
- Shape manipulation (`reshape`, `transpose`)

#### Additional Libraries (Helpful)
- **Matplotlib/Seaborn**: For visualizing results
- **Pandas**: For data manipulation in examples
- **Jupyter Notebooks**: For interactive learning

#### Recommended Python Resources
- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

## 📁 Repository Structure

```
ml-from-scratch/
│
├── models/
│   ├── linear_models/
│   │   ├── linear_regression.py
│   │   ├── ridge_regression.py
│   │   └── logistic_regression.py
│   │
│   ├── tree_models/          # Coming soon
│   ├── ensemble_models/      # Coming soon
│   ├── svm_models/           # Coming soon
│   └── clustering_models/    # Coming soon
│
├── tutorials/
│   ├── linear_regression_math.md
│   ├── logistic_regression_math.md
│   ├── ridge_regression_math.md
│   └── ...
│
├── notebooks/
│   ├── 01_linear_regression_demo.ipynb
│   ├── 02_logistic_regression_demo.ipynb
│   └── ...
│
├── datasets/                 # Links and loading scripts
├── tests/                    # Unit tests for each model
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/TakundaDanha/ml-from-scratch.git
cd ml-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Example

```python
import numpy as np
from models.linear_models.linear_regression import LinearRegression

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Initialize and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")
print(f"Parameters: {model.get_params()}")
```

## 📊 Datasets Used

- Will update this section once I've completed more models

## 📝 Tutorials

### Mathematical Foundations

#### Linear Models
- [Linear Regression - Mathematical Foundation](tutorials/linear_regression_math.md)
- [Ridge Regression - Mathematical Foundation](tutorials/ridge_regression_math.md)
- [Logistic Regression - Derivation](tutorials/logistic_regression_math.md)

#### Tree Models
- [Decision Trees - Mathematical Foundation](tutorials/decision_tree_math.md)
- [Regression Trees - Mathematical Foundation](tutorials/regression_tree_math.md)
- [Random Forests - Mathematical Foundation](tutorials/random_forests_math.md)

### Models from scratch
#### Linear Models
- [Linear Regression](models/linear_models/linear_regression.py)
- [Ridge Regression](modesls/linear_models/ridge_regression.py)
- [Logistic Regression](models/linear_models/logistic_regression.py)

#### Tree Models
- [Decision Trees](models/tree_models/decision_tree.py)
- [Regression Trees](models/tree_models/regression_tree.py)
- [Random Forests](models/tree_models/random_forest.py)

### Jupyter Notebooks

#### Linear Models
- [Linear/Ridge Regression Demo](notebooks/01_linear_regression_demo.ipynb)
- [Logistic Regression Demo](notebooks/02_logistic_regression_demo.ipynb)

#### Tree Models
- [Decision Trees Demo](notebooks/03_decision_trees_demo.ipynb)
- [Regression Trees Demo](notebooks/04_regression_trees_demo.ipynb)
- [Random Forests Demo](notebooks/05_random_forests_demo.ipynb)

## 🎓 Learning Path

### Beginner Path
1. Start with **Linear Regression** - simplest algorithm
2. Move to **Logistic Regression** - introduces classification
3. Learn **Ridge Regression** - introduces regularization
4. Try **K-Nearest Neighbors** - different paradigm

### Intermediate Path
1. **Decision Trees** - non-linear decision boundaries
2. **Random Forests** - ensemble learning
3. **Support Vector Machines** - margin optimization
4. **Naive Bayes** - probabilistic approach

### Advanced Path
1. **Neural Networks** - deep learning foundations
2. **Principal Component Analysis** - dimensionality reduction
3. **Gradient Boosting** - advanced ensembles

## 🤝 Contributing

This is an educational project, and contributions are welcome! Whether it's:
- Fixing typos or errors in math derivations
- Adding new models
- Improving code documentation
- Creating additional examples
- Suggesting better explanations

Please feel free to open an issue or submit a pull request.

## 📜 License

Feel free to use this for learning, teaching, or any other purpose.

## 🙏 Acknowledgments

This project is inspired by the need for clear, step-by-step explanations of machine learning algorithms that bridge the gap between theory and implementation.

## 📧 Contact

Questions? Suggestions? Feel free to open an issue or reach out!

---

**Note**: This is a work in progress. Check back regularly for new models and tutorials!