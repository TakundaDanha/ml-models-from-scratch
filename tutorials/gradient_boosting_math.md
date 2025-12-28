# Gradient Boosting - Mathematical Modelling

## Overview
Gradient boosting is an ensemble learning technique that builds a strong predictive model by sequentially combining multiple weak learners, typically decision trees. Unlike random forests, which train trees independently in parallel and aggregate their outputs, gradient boosting trains each new model to correct the errors made by the previous ones. This is achieved by optimizing a specified loss function using gradient descent in function space.

At each iteration, a new model is fitted to the negative gradient (pseudo-residuals) of the loss function with respect to the current ensemble's predictions. By gradually minimizing the loss, gradient boosting produces highly accurate models capable of capturing complex nonlinear relationships. While it often achieves lower bias than bagging-based methods, it is more sensitive to overfitting and requires careful tuning of hyperparameters such as learning rate, tree depth, and number of estimators.

---

## Key Intuition: Gradient Descent in Function Space

Traditional gradient descent optimizes parameters:
**θ_new = θ_old - α × ∇L(θ)**

Gradient boosting optimizes the prediction function itself:
**F_m(x) = F_{m-1}(x) - α × ∇L(F)**

Instead of adjusting weights, we add new functions (trees) that point in the direction of steepest descent of the loss function.

---

## Additive Modelling

Gradient boosting builds a model as a weighted sum of base learners (usually shallow trees called "weak learners"):

**F_M(x) = F_0(x) + γ_1·h_1(x) + γ_2·h_2(x) + ... + γ_M·h_M(x)**

Where:
- **F_M(x)** is the final ensemble prediction
- **F_0(x)** is the initial model (usually a constant)
- **h_m(x)** are the base learners (decision trees)
- **γ_m** are the weights (step sizes) for each tree
- **M** is the total number of boosting iterations

Each new term **γ_m·h_m(x)** is added to reduce the loss function, moving us closer to the optimal prediction.

---

## Loss Functions

Gradient boosting can optimize different loss functions depending on the problem:

### Regression:
1. **Mean Squared Error (MSE):**
   - L(y, F(x)) = (1/2)(y - F(x))²
   - Derivative: -∂L/∂F = y - F(x) (the residual)

2. **Mean Absolute Error (MAE):**
   - L(y, F(x)) = |y - F(x)|
   - Derivative: -∂L/∂F = sign(y - F(x))

3. **Huber Loss** (robust to outliers):
   - Combines MSE and MAE properties

### Classification (Binary):
1. **Log Loss (Logistic/Bernoulli):**
   - L(y, F(x)) = log(1 + exp(-2yF(x))) where y ∈ {-1, +1}
   - Used in most implementations for binary classification

2. **Exponential Loss:**
   - L(y, F(x)) = exp(-yF(x))
   - Used in AdaBoost

---

## Gradient Boosting Algorithm

### Input:
- Training set {(x_i, y_i)}, i = 1, ..., n
- A differentiable loss function L(y, F(x))
- Number of iterations M
- Learning rate η (shrinkage parameter)

### Algorithm:

**Step 1: Initialize the model with a constant value**

F_0(x) = arg min_γ Σ L(y_i, γ)

For MSE regression: F_0(x) = mean(y)
For classification: F_0(x) = log(odds) = log(p/(1-p))

---

**Step 2: For m = 1 to M (boosting iterations):**

**2.1) Compute pseudo-residuals (negative gradient):**

For each training sample i = 1, ..., n:

r_{im} = -[∂L(y_i, F(x_i))/∂F(x_i)]|_{F(x)=F_{m-1}(x)}

These pseudo-residuals represent the direction of steepest descent of the loss function. They tell us what corrections we need to make to improve our current predictions.

**For MSE:** r_{im} = y_i - F_{m-1}(x_i) (actual residuals)
**For other losses:** r_{im} varies based on the derivative

---

**2.2) Fit a base learner to pseudo-residuals:**

Train a regression tree h_m(x) on the training set {(x_i, r_{im})}, i = 1, ..., n

Key points:
- The tree learns to predict the pseudo-residuals, not the original targets
- We typically use shallow trees (depth 3-8) to avoid overfitting
- Each leaf of h_m(x) contains a constant prediction value

---

**2.3) Compute optimal leaf values (terminal regions):**

For each leaf region R_{jm} in tree h_m:

γ_{jm} = arg min_γ Σ_{x_i ∈ R_{jm}} L(y_i, F_{m-1}(x_i) + γ)

This step finds the best constant value to predict in each leaf to minimize the loss.

**For MSE:** γ_{jm} = mean(r_{im}) for samples in leaf R_{jm}
**For other losses:** requires line search or closed-form solution

---

**2.4) Update the model with learning rate:**

F_m(x) = F_{m-1}(x) + η · h_m(x)

Where:
- **η** is the learning rate (typically 0.01 to 0.3)
- Smaller η requires more iterations M but often generalizes better
- This is called "shrinkage" and helps prevent overfitting

---

**Step 3: Output F_M(x)**

The final model is the sum of all M boosted trees:

F_M(x) = F_0(x) + η·Σ h_m(x) for m = 1 to M

---

## Hyperparameters

### Essential Hyperparameters:

1. **n_estimators (M):** Number of boosting iterations
   - More trees = better training fit but risk of overfitting
   - Typical: 100-1000
   - Use early stopping with validation set

2. **learning_rate (η):** Shrinkage parameter
   - Controls contribution of each tree
   - Typical: 0.01-0.3
   - Trade-off: small η needs large M

3. **max_depth:** Maximum depth of each tree
   - Typical: 3-8 (shallow trees work best)
   - Deeper trees capture more interactions but overfit easier
   - Depth 1 = "stumps" (no interactions)

4. **min_samples_split:** Minimum samples to split a node
   - Typical: 2-20
   - Higher values prevent overfitting

5. **min_samples_leaf:** Minimum samples in leaf nodes
   - Typical: 1-20
   - Higher values create smoother predictions

6. **subsample:** Fraction of samples for each tree
   - Typical: 0.5-1.0
   - < 1.0 introduces randomness (Stochastic Gradient Boosting)
   - Helps prevent overfitting and speeds up training

### Optional Hyperparameters:

1. **max_features:** Number of features to consider for splits
   - Similar to random forests
   - Typical: None (use all), sqrt(n_features), log2(n_features)

2. **loss:** Loss function to optimize
   - Regression: 'squared_error', 'absolute_error', 'huber'
   - Classification: 'log_loss', 'exponential'

3. **validation_fraction:** For early stopping
   - Typical: 0.1-0.2 of training data

---

## Complete Example: Predicting House Prices with Gradient Boosting

Consider the same dataset of 12 houses:

| House | Location | Size   | Price |
|-------|----------|--------|-------|
| 1     | Urban    | Small  | 150   |
| 2     | Urban    | Small  | 160   |
| 3     | Urban    | Medium | 200   |
| 4     | Urban    | Medium | 210   |
| 5     | Urban    | Large  | 280   |
| 6     | Suburban | Small  | 140   |
| 7     | Suburban | Small  | 145   |
| 8     | Suburban | Medium | 190   |
| 9     | Suburban | Large  | 250   |
| 10    | Rural    | Medium | 170   |
| 11    | Rural    | Large  | 220   |
| 12    | Rural    | Large  | 230   |

We'll build a gradient boosting model with:
- **M = 3 iterations** (trees)
- **Learning rate η = 0.3**
- **Loss function: MSE**
- **Max depth = 2** (for illustration)

---

### Step 1: Initialize F_0(x)

For MSE regression, initialize with the mean of all target values:

F_0(x) = mean(150, 160, 200, 210, 280, 140, 145, 190, 250, 170, 220, 230)
F_0(x) = **195.42**

Initial predictions for all houses: **195.42**

---

### Iteration 1: Building First Tree (m=1)

**Step 2.1: Compute pseudo-residuals**

For MSE: r_{i1} = y_i - F_0(x_i)

| House | y_i | F_0(x_i) | Residual r_{i1} |
|-------|-----|----------|-----------------|
| 1     | 150 | 195.42   | -45.42          |
| 2     | 160 | 195.42   | -35.42          |
| 3     | 200 | 195.42   | 4.58            |
| 4     | 210 | 195.42   | 14.58           |
| 5     | 280 | 195.42   | 84.58           |
| 6     | 140 | 195.42   | -55.42          |
| 7     | 145 | 195.42   | -50.42          |
| 8     | 190 | 195.42   | -5.42           |
| 9     | 250 | 195.42   | 54.58           |
| 10    | 170 | 195.42   | -25.42          |
| 11    | 220 | 195.42   | 24.58           |
| 12    | 230 | 195.42   | 34.58           |

---

**Step 2.2: Fit tree h_1(x) to residuals**

We train a regression tree (max_depth=2) to predict the residuals.

**Root node:** All 12 samples, mean residual = 0

Try splitting on **Size**:
- **Small:** {1, 2, 6, 7} → mean residual = (-45.42 - 35.42 - 55.42 - 50.42)/4 = **-46.67**
- **Medium:** {3, 4, 8, 10} → mean residual = (4.58 + 14.58 - 5.42 - 25.42)/4 = **-2.92**
- **Large:** {5, 9, 11, 12} → mean residual = (84.58 + 54.58 + 24.58 + 34.58)/4 = **49.58**

This split has good separation! Let's use it.

Now at depth 2, we can split each node once more:

**For Small houses:** Split by Location:
- Urban {1, 2}: mean residual = (-45.42 - 35.42)/2 = **-40.42**
- Suburban {6, 7}: mean residual = (-55.42 - 50.42)/2 = **-52.92**

**For Medium houses:** Split by Location:
- Urban {3, 4}: mean residual = (4.58 + 14.58)/2 = **9.58**
- Suburban {8}: mean residual = **-5.42**
- Rural {10}: mean residual = **-25.42**

Since we can only split into 2 groups, let's split Medium as:
- Urban {3, 4}: **9.58**
- Suburban+Rural {8, 10}: (-5.42 - 25.42)/2 = **-15.42**

**For Large houses:** Split by Location:
- Urban {5}: mean residual = **84.58**
- Suburban {9}: mean residual = **54.58**
- Rural {11, 12}: mean residual = (24.58 + 34.58)/2 = **29.58**

Split Large as:
- Urban {5}: **84.58**
- Suburban+Rural {9, 11, 12}: (54.58 + 24.58 + 34.58)/3 = **37.91**

**Tree 1 Structure (h_1(x)):**
```
Root: Size
├── Small
│   ├── Urban → Leaf: -40.42
│   └── Suburban → Leaf: -52.92
├── Medium
│   ├── Urban → Leaf: 9.58
│   └── Suburban+Rural → Leaf: -15.42
└── Large
    ├── Urban → Leaf: 84.58
    └── Suburban+Rural → Leaf: 37.91
```

---

**Step 2.4: Update model**

F_1(x) = F_0(x) + η · h_1(x) = F_0(x) + 0.3 · h_1(x)

Let's calculate F_1(x) for each house:

| House | F_0(x) | h_1(x)  | 0.3·h_1(x) | F_1(x)  |
|-------|--------|---------|------------|---------|
| 1     | 195.42 | -40.42  | -12.13     | 183.29  |
| 2     | 195.42 | -40.42  | -12.13     | 183.29  |
| 3     | 195.42 | 9.58    | 2.87       | 198.29  |
| 4     | 195.42 | 9.58    | 2.87       | 198.29  |
| 5     | 195.42 | 84.58   | 25.37      | 220.79  |
| 6     | 195.42 | -52.92  | -15.88     | 179.54  |
| 7     | 195.42 | -52.92  | -15.88     | 179.54  |
| 8     | 195.42 | -15.42  | -4.63      | 190.79  |
| 9     | 195.42 | 37.91   | 11.37      | 206.79  |
| 10    | 195.42 | -15.42  | -4.63      | 190.79  |
| 11    | 195.42 | 37.91   | 11.37      | 206.79  |
| 12    | 195.42 | 37.91   | 11.37      | 206.79  |

---

### Iteration 2: Building Second Tree (m=2)

**Step 2.1: Compute new residuals**

r_{i2} = y_i - F_1(x_i)

| House | y_i | F_1(x_i) | Residual r_{i2} |
|-------|-----|----------|-----------------|
| 1     | 150 | 183.29   | -33.29          |
| 2     | 160 | 183.29   | -23.29          |
| 3     | 200 | 198.29   | 1.71            |
| 4     | 210 | 198.29   | 11.71           |
| 5     | 280 | 220.79   | 59.21           |
| 6     | 140 | 179.54   | -39.54          |
| 7     | 145 | 179.54   | -34.54          |
| 8     | 190 | 190.79   | -0.79           |
| 9     | 250 | 206.79   | 43.21           |
| 10    | 170 | 190.79   | -20.79          |
| 11    | 220 | 206.79   | 13.21           |
| 12    | 230 | 206.79   | 23.21           |

Notice the residuals are smaller now! The model is improving.

---

**Step 2.2: Fit tree h_2(x) to new residuals**

Following similar logic with max_depth=2, we'd build another tree. For brevity, let's say h_2(x) produces these predictions:

Tree 2 learns to correct remaining errors with leaves predicting values based on the new residual patterns.

**Step 2.4: Update model**

F_2(x) = F_1(x) + 0.3 · h_2(x)

---

### Iteration 3: Building Third Tree (m=3)

**Step 2.1: Compute residuals** r_{i3} = y_i - F_2(x_i)

**Step 2.2: Fit tree h_3(x)**

**Step 2.4: Final update**

F_3(x) = F_2(x) + 0.3 · h_3(x)

---

## Making Predictions with Gradient Boosting

For a new house with **Location=Urban, Size=Large**:

**Step 1:** Start with F_0(x) = 195.42

**Step 2:** Add contribution from Tree 1:
- h_1(x) = 84.58 (Urban, Large leaf)
- Add: 0.3 × 84.58 = 25.37
- Running total: 195.42 + 25.37 = 220.79

**Step 3:** Add contribution from Tree 2:
- h_2(x) = [some value based on tree structure]
- Add: 0.3 × h_2(x)

**Step 4:** Add contribution from Tree 3:
- h_3(x) = [some value]
- Add: 0.3 × h_3(x)

**Final Prediction:** F_3(x) = Sum of all contributions

Each tree corrects the mistakes of the previous ensemble, gradually refining the prediction.

---

## Gradient Boosting vs Random Forests

### Key Differences:

| Aspect | Random Forests | Gradient Boosting |
|--------|----------------|-------------------|
| **Training** | Parallel (independent trees) | Sequential (each tree depends on previous) |
| **Tree depth** | Deep, unpruned trees | Shallow trees (stumps to depth 8) |
| **Prediction** | Average/vote of all trees | Weighted sum of trees |
| **Error correction** | None (independent) | Each tree corrects previous errors |
| **Speed** | Fast (parallelizable) | Slower (sequential) |
| **Overfitting risk** | Low (due to averaging) | Higher (requires regularization) |
| **Variance** | Low (bagging reduces it) | Can be high without proper tuning |
| **Bias** | Moderate | Low (powerful) |
| **Interpretability** | Low | Very low |
| **Hyperparameter sensitivity** | Low | High |

### When to Use Each:

**Random Forests:**
- When you want a robust baseline with minimal tuning
- When you can tolerate moderate bias for stability
- When training time should be fast
- When overfitting is a major concern

**Gradient Boosting:**
- When you need the best possible accuracy
- When you have time for careful hyperparameter tuning
- When you have sufficient data to prevent overfitting
- When bias reduction is more important than variance

---

## Regularization Techniques

Gradient boosting is prone to overfitting. Here are techniques to prevent it:

### 1. Shrinkage (Learning Rate)
Use small learning rate η (e.g., 0.01-0.1) with more trees:

**Effect:** Smaller steps mean slower learning but better generalization

**Trade-off:** Lower η requires more iterations M

### 2. Tree Constraints
- **max_depth:** Limit tree depth (typically 3-8)
- **min_samples_split:** Require minimum samples to split
- **min_samples_leaf:** Require minimum samples in leaves
- **max_leaf_nodes:** Limit total number of leaves

### 3. Stochastic Gradient Boosting
Subsample training data for each tree (subsample < 1.0):

**subsample = 0.5** means each tree uses random 50% of data

**Benefits:**
- Adds randomness (like bagging)
- Speeds up training
- Reduces overfitting

### 4. Feature Subsampling
Similar to random forests, use only subset of features per split:

**max_features = 'sqrt'** or **'log2'**

### 5. Early Stopping
Monitor validation loss and stop when it stops improving:

```python
n_iter_no_change = 10  # Stop if no improvement for 10 iterations
validation_fraction = 0.1  # Use 10% for validation
```

### 6. Regularization Terms
Some implementations (XGBoost, LightGBM) add L1/L2 penalties to leaf weights.

---

## Advanced Gradient Boosting Variants

### 1. XGBoost (Extreme Gradient Boosting)
**Improvements:**
- Regularized loss function with L1/L2 penalties
- Second-order Taylor approximation of loss
- Efficient tree construction algorithm
- Built-in handling of missing values
- Parallel processing for tree construction
- Hardware optimization

**Loss function:**
Obj = Σ L(y_i, F(x_i)) + Σ Ω(h_m)

Where Ω(h_m) = γT + (λ/2)||w||² (regularization on tree structure)

### 2. LightGBM (Light Gradient Boosting Machine)
**Improvements:**
- Histogram-based algorithm (faster)
- Leaf-wise tree growth (vs level-wise)
- Gradient-based One-Side Sampling (GOSS)
- Exclusive Feature Bundling (EFB)
- Supports categorical features directly

**Leaf-wise growth:** Splits the leaf with maximum gain, creating deeper, more asymmetric trees

### 3. CatBoost (Categorical Boosting)
**Improvements:**
- Ordered boosting (reduces prediction shift)
- Native handling of categorical features
- Symmetric trees (balanced structure)
- Robust to overfitting

### 4. HistGradientBoosting (scikit-learn)
**Improvements:**
- Histogram-based (like LightGBM)
- Fast on large datasets
- Native missing value support
- Monotonic constraints

---

## Feature Importance

Gradient boosting can measure feature importance:

### 1. Gain-Based Importance
For each feature, sum the gain (reduction in loss) from all splits using that feature across all trees:

**Importance(Feature_j) = Σ_{m=1}^M Σ_{splits on j} Gain**

### 2. Split-Based Importance
Count how many times each feature is used for splitting:

**Importance(Feature_j) = Number of splits using Feature_j**

### 3. Permutation Importance
1. Calculate model's loss on validation set
2. Randomly permute values of Feature_j
3. Recalculate loss
4. Importance = Increase in loss

**Higher importance = more critical feature**

---

## Practical Guidelines

### 1. Starting Hyperparameters:
```python
n_estimators = 100
learning_rate = 0.1
max_depth = 3
min_samples_split = 2
min_samples_leaf = 1
subsample = 1.0
```

### 2. Tuning Strategy:
1. **First:** Fix learning_rate=0.1, tune tree parameters (max_depth, min_samples_leaf)
2. **Second:** Fix tree parameters, tune n_estimators with early stopping
3. **Third:** Lower learning_rate (0.01-0.05) and increase n_estimators
4. **Fourth:** Add stochastic elements (subsample, max_features)
5. **Fifth:** Fine-tune remaining parameters

### 3. Learning Rate vs Number of Estimators:
- **η = 0.1, M = 100:** Good starting point
- **η = 0.01, M = 1000:** Better generalization, slower training
- **η = 0.3, M = 50:** Fast but may overfit

**Rule of thumb:** η × M ≈ constant (trade-off between speed and performance)

### 4. Validation Strategy:
- Use holdout validation set for early stopping
- Monitor both training and validation loss
- Stop when validation loss increases for N consecutive iterations

### 5. Diagnosing Problems:
- **High training error:** Increase model complexity (more trees, deeper trees)
- **Large train-validation gap:** Reduce complexity, increase regularization
- **Slow convergence:** Increase learning rate or tree depth
- **Unstable predictions:** Increase subsample ratio, reduce learning rate

---

## Mathematical Properties

### Convergence
For convex loss functions and appropriate step sizes, gradient boosting converges to the optimal solution:

**lim_{M→∞} F_M(x) = F*(x)**

Where F*(x) minimizes the expected loss E[L(y, F(x))]

### Bias-Variance Trade-off
- **Bias:** Decreases with each iteration (each tree reduces systematic errors)
- **Variance:** Can increase with iterations (overfitting to training noise)

**Optimal M:** Found by monitoring validation error

### Connection to Gradient Descent
Each iteration performs a functional gradient descent step:

**F_m = F_{m-1} - η · ∇L(F_{m-1})**

But instead of computing the exact gradient, we:
1. Compute gradient at each training point (pseudo-residuals)
2. Fit a tree to approximate this gradient
3. Move in the direction suggested by the tree

This is why it's called "gradient" boosting!

---

## Advantages of Gradient Boosting

1. **High Predictive Accuracy:** Often wins Kaggle competitions
2. **Flexible:** Works with any differentiable loss function
3. **Feature Interactions:** Automatically captures complex interactions
4. **Handles Mixed Data:** Works with numerical and categorical features
5. **Robust to Outliers:** Especially with robust loss functions (Huber, MAE)
6. **Feature Importance:** Built-in feature ranking
7. **Handles Missing Values:** Advanced implementations do this well
8. **No Feature Scaling Required:** Tree-based, invariant to monotonic transformations

---

## Disadvantages of Gradient Boosting

1. **Computationally Expensive:** Sequential training (not parallelizable across trees)
2. **Hyperparameter Sensitive:** Requires careful tuning
3. **Prone to Overfitting:** Needs regularization and validation
4. **Less Interpretable:** Ensemble of trees is hard to understand
5. **Memory Intensive:** Stores all trees in memory
6. **Not Ideal for Extrapolation:** Like all trees, cannot predict beyond training range
7. **Sensitive to Noisy Data:** Can overfit to noise without proper regularization
8. **Training Time:** Longer than random forests for same number of trees

---

## Comparison Example: Single Tree vs Random Forest vs Gradient Boosting

For the house price dataset:

**Single Decision Tree:**
- Prediction for Urban, Large: 245
- Simple, fast, interpretable
- High variance, potential overfitting

**Random Forest (3 trees):**
- Prediction for Urban, Large: 228.17
- Averages independent trees
- Reduced variance, stable predictions
- Parallel training

**Gradient Boosting (3 iterations):**
- Prediction for Urban, Large: ~220-230 (after 3 iterations)
- Sequential error correction
- Can achieve lower bias with more iterations
- Requires careful tuning to avoid overfitting

---

## When Different Loss Functions Are Used

### Mean Squared Error (MSE):
- Standard regression
- Sensitive to outliers
- Produces smooth predictions

### Mean Absolute Error (MAE):
- Robust regression
- Less sensitive to outliers
- Predicts median rather than mean

### Huber Loss:
- Combines MSE and MAE
- Smooth for small errors, linear for large errors
- Balanced approach

### Log Loss (Classification):
- Binary and multiclass classification
- Produces probability estimates
- Penalizes confident wrong predictions heavily

---

## Conclusion

Gradient boosting is a powerful ensemble method that:
- Builds models sequentially, with each new model correcting errors of previous ones
- Optimizes a loss function through functional gradient descent
- Achieves low bias through iterative refinement
- Requires careful regularization to prevent overfitting
- Offers state-of-the-art performance with proper tuning

The key to success with gradient boosting is understanding the bias-variance trade-off and using appropriate regularization techniques. While it requires more tuning than random forests, the potential for superior accuracy makes it a go-to choice for many machine learning competitions and production systems.

**Core principle:** Start with a simple model, identify its mistakes, train a new model to fix those mistakes, and repeat. Each iteration makes a small improvement, and the cumulative effect is a highly accurate ensemble.