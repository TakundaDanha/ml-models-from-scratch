# Random Forests - Mathematical Modelling

## Overview
Random forests are an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. The technique works by averaging predictions from multiple decision trees, each trained on different random subsets of the data and features. This approach reduces variance and overfitting while maintaining low bias, generally resulting in superior performance compared to individual decision trees.

---

## Algorithm

Given a dataset **D** with **n** datapoints and **m** features, a random forest with **B** trees is created as follows:

**For each tree b = 1, 2, ..., B:**

1. **Bootstrap Sampling:** Create a bootstrap sample **D_b** by randomly selecting **n** datapoints from **D** with replacement. This means some datapoints may appear multiple times while others may not appear at all.

2. **Feature Subset Selection:** At each node during tree construction, randomly select **k** features from the total **m** features (typically k = √m for classification or k = m/3 for regression).

3. **Tree Construction:** Build a decision tree using only the selected features, without pruning. Split nodes using the best feature among the **k** randomly selected features.

4. **Repeat:** Continue until all **B** trees are constructed.

---

## Key Concepts

### Bootstrap Aggregating (Bagging)
Bootstrap sampling with replacement ensures that:
- On average, each bootstrap sample contains about 63.2% of the unique datapoints from the original dataset
- About 36.8% of datapoints are left out (these become the Out-of-Bag samples)
- Each tree sees a slightly different version of the data, promoting diversity

### Feature Randomness
By limiting each split to consider only **k** randomly selected features:
- Trees are decorrelated from each other
- No single strong predictor can dominate all trees
- The forest captures different patterns and relationships in the data

### Diversity Principle
Each tree in the forest should be unique due to:
- Different bootstrap samples
- Different feature subsets at each split
- This diversity reduces overfitting and improves generalization

---

## Hyperparameters

### Essential Hyperparameters:
1. **n_estimators (B):** Number of trees in the forest (typical: 100-500)
2. **max_features (k):** Number of features to consider at each split
   - Classification: √m (square root of total features)
   - Regression: m/3 (one-third of total features)
3. **bootstrap:** Whether to use bootstrap sampling (typically True)
4. **criterion:** Splitting criterion
   - Classification: Gini impurity or Entropy
   - Regression: MSE (Mean Squared Error)

### Optional Hyperparameters:
1. **max_depth:** Maximum depth of each tree (default: None, grow until pure)
2. **min_samples_split:** Minimum samples required to split an internal node
3. **min_samples_leaf:** Minimum samples required at a leaf node
4. **max_leaf_nodes:** Maximum number of leaf nodes per tree

---

## Making Predictions

### For Classification:
For a given input **x**, generate predictions from all **B** trees and use **majority voting**:

**ŷ = mode{T₁(x), T₂(x), ..., T_B(x)}**

### For Regression:
For a given input **x**, generate predictions from all **B** trees and **average** them:

**ŷ = (1/B) × Σ Tᵢ(x)** for i = 1 to B

---

## Out-of-Bag (OOB) Error Estimation

Random forests have a built-in validation mechanism using Out-of-Bag samples.

### OOB Error Calculation:

For each datapoint **xᵢ** in the original dataset:

1. **Identify OOB trees:** Find all trees that were NOT trained on **xᵢ** (approximately 36.8% of trees)

2. **Make OOB prediction:** Use only these OOB trees to predict **xᵢ**
   - Classification: ŷᵢ^(OOB) = majority vote from OOB trees
   - Regression: ŷᵢ^(OOB) = average prediction from OOB trees

3. **Calculate OOB error for xᵢ:**
   - Classification: error_i = 1 if ŷᵢ^(OOB) ≠ yᵢ, else 0
   - Regression: error_i = (ŷᵢ^(OOB) - yᵢ)²

4. **Overall OOB error:**

**OOB_error = (1/n) × Σ error_i** for all datapoints

The OOB error provides an unbiased estimate of the generalization error without requiring a separate validation set.

---

## Complete Example: Predicting House Prices with Random Forest

Consider the same dataset of 12 houses:

| House | Location | Size    | Price |
|-------|----------|---------|-------|
| 1     | Urban    | Small   | 150   |
| 2     | Urban    | Small   | 160   |
| 3     | Urban    | Medium  | 200   |
| 4     | Urban    | Medium  | 210   |
| 5     | Urban    | Large   | 280   |
| 6     | Suburban | Small   | 140   |
| 7     | Suburban | Small   | 145   |
| 8     | Suburban | Medium  | 190   |
| 9     | Suburban | Large   | 250   |
| 10    | Rural    | Medium  | 170   |
| 11    | Rural    | Large   | 220   |
| 12    | Rural    | Large   | 230   |

We'll build a random forest with **B = 3 trees** and **k = 1 feature per split** (simplified for illustration).

---

### Tree 1: Bootstrap Sample 1

**Bootstrap Sample:** Randomly select 12 houses with replacement:
{1, 1, 3, 4, 5, 7, 8, 9, 10, 11, 11, 12}

**OOB samples for Tree 1:** {2, 6} (not selected)

**Building Tree 1:**

At root node, we have 12 samples (with house 1 and 11 appearing twice).

**Step 1:** Randomly select k=1 feature. Let's say we select **Size**.

Calculate MSE and Gain for Size:
- Original sample mean: μ = (150 + 150 + 200 + 210 + 280 + 145 + 190 + 250 + 170 + 220 + 220 + 230)/12 = 201.25
- MSE(S) = 1625.94 (calculated as in regression tree example)

Split by Size:
- **Small:** {1, 1, 7} → Prices: [150, 150, 145] → μ = 148.33
- **Medium:** {3, 4, 8, 10} → Prices: [200, 210, 190, 170] → μ = 192.5
- **Large:** {5, 9, 11, 11, 12} → Prices: [280, 250, 220, 220, 230] → μ = 240

**Tree 1 Structure:**
```
Root: Size
├── Small → Leaf: 148.33
├── Medium → Leaf: 192.5
└── Large → Leaf: 240
```

---

### Tree 2: Bootstrap Sample 2

**Bootstrap Sample:** Randomly select 12 houses with replacement:
{2, 3, 3, 4, 6, 7, 8, 8, 9, 10, 11, 12}

**OOB samples for Tree 2:** {1, 5} (not selected)

**Building Tree 2:**

At root node, randomly select k=1 feature. Let's say we select **Location**.

Split by Location:
- **Urban:** {2, 3, 3, 4} → Prices: [160, 200, 200, 210] → μ = 192.5
- **Suburban:** {6, 7, 8, 8, 9} → Prices: [140, 145, 190, 190, 250] → μ = 183
- **Rural:** {10, 11, 12} → Prices: [170, 220, 230] → μ = 206.67

**Tree 2 Structure:**
```
Root: Location
├── Urban → Leaf: 192.5
├── Suburban → Leaf: 183
└── Rural → Leaf: 206.67
```

---

### Tree 3: Bootstrap Sample 3

**Bootstrap Sample:** Randomly select 12 houses with replacement:
{1, 2, 4, 5, 5, 6, 7, 9, 10, 10, 11, 12}

**OOB samples for Tree 3:** {3, 8} (not selected)

**Building Tree 3:**

At root node, randomly select k=1 feature. Let's say we select **Size**.

Split by Size:
- **Small:** {1, 2, 6, 7} → Prices: [150, 160, 140, 145] → μ = 148.75
- **Medium:** {4, 10, 10} → Prices: [210, 170, 170] → μ = 183.33
- **Large:** {5, 5, 9, 11, 12} → Prices: [280, 280, 250, 220, 230] → μ = 252

**Tree 3 Structure:**
```
Root: Size
├── Small → Leaf: 148.75
├── Medium → Leaf: 183.33
└── Large → Leaf: 252
```

---

## Making Predictions with the Random Forest

### Example 1: Predict price for a new house with Location=Urban, Size=Large

**Tree 1:** Size=Large → Prediction: **240**
**Tree 2:** Location=Urban → Prediction: **192.5**
**Tree 3:** Size=Large → Prediction: **252**

**Random Forest Prediction:**
ŷ = (240 + 192.5 + 252) / 3 = **228.17**

(Note: The single tree with Size as root predicted 245 for Large houses. The random forest provides a different estimate by incorporating multiple perspectives.)

---

### Example 2: Predict price for Location=Suburban, Size=Small

**Tree 1:** Size=Small → Prediction: **148.33**
**Tree 2:** Location=Suburban → Prediction: **183**
**Tree 3:** Size=Small → Prediction: **148.75**

**Random Forest Prediction:**
ŷ = (148.33 + 183 + 148.75) / 3 = **160.03**

---

## Calculating OOB Error

For each house, we use only the trees where it was OOB (not in the bootstrap sample).

**House 1:** OOB for Trees 2 and 3
- Tree 2: Urban → 192.5
- Tree 3: Small → 148.75
- OOB prediction: (192.5 + 148.75)/2 = **170.63**
- True value: 150
- Error₁ = (170.63 - 150)² = **425.64**

**House 2:** OOB for Tree 1
- Tree 1: Small → 148.33
- OOB prediction: **148.33**
- True value: 160
- Error₂ = (148.33 - 160)² = **136.13**

**House 3:** OOB for Tree 3
- Tree 3: Medium → 183.33
- OOB prediction: **183.33**
- True value: 200
- Error₃ = (183.33 - 200)² = **277.78**

**House 5:** OOB for Tree 2
- Tree 2: Urban → 192.5
- OOB prediction: **192.5**
- True value: 280
- Error₅ = (192.5 - 280)² = **7656.25**

**House 6:** OOB for Tree 1
- Tree 1: Small → 148.33
- OOB prediction: **148.33**
- True value: 140
- Error₆ = (148.33 - 140)² = **69.39**

**House 8:** OOB for Tree 3
- Tree 3: Medium → 183.33
- OOB prediction: **183.33**
- True value: 190
- Error₈ = (183.33 - 190)² = **44.49**

**OOB Error (for houses with OOB predictions):**
OOB_MSE = (425.64 + 136.13 + 277.78 + 7656.25 + 69.39 + 44.49) / 6 = **1434.95**

This OOB error estimate gives us an idea of how well the random forest generalizes without needing a separate validation set.

---

## Advantages of Random Forests

1. **Reduced Overfitting:** Averaging multiple trees reduces variance significantly
2. **Handles High Dimensionality:** Works well even with many features
3. **Feature Importance:** Can rank features by their contribution to predictions
4. **Robust to Outliers:** Individual outliers have less impact when averaged
5. **No Need for Feature Scaling:** Tree-based methods are invariant to monotonic transformations
6. **Built-in Validation:** OOB error provides unbiased performance estimate
7. **Handles Missing Values:** Can maintain accuracy even with missing data
8. **Parallel Training:** Each tree can be built independently

---

## Disadvantages of Random Forests

1. **Less Interpretable:** Harder to explain than a single decision tree
2. **Computational Cost:** Training and prediction are slower than single trees
3. **Memory Usage:** Storing B trees requires more memory
4. **Not Ideal for Extrapolation:** Cannot predict beyond the range of training data
5. **Biased Toward Categorical Features:** Features with more categories may be preferred
6. **Incremental Learning:** Difficult to update with new data (requires retraining)

---

## Feature Importance

Random forests can measure feature importance in two ways:

### 1. Mean Decrease in Impurity (MDI)
For each feature, average the total reduction in MSE (or Gini/Entropy) across all trees where that feature was used for splitting.

**Importance(Feature_j) = (1/B) × Σ Σ (Gain achieved by Feature_j in Tree_i)**

### 2. Mean Decrease in Accuracy (MDA) - Permutation Importance
1. Calculate OOB error
2. Randomly permute values of Feature_j
3. Recalculate OOB error with permuted feature
4. Importance = Increase in OOB error

Features that cause large increases in error when permuted are more important.

---

## Comparison: Single Tree vs Random Forest

Using our example:

**Single Regression Tree (from regression_tree_math.md):**
- Root split on Size
- Prediction for Large houses: 245

**Random Forest (3 trees):**
- Prediction for Urban, Large: 228.17
- Uses both Size and Location information
- More robust to the specific sample

The random forest provides predictions that are less sensitive to the training data and typically generalize better to new data.

---

## Mathematical Properties

### Variance Reduction
For B independent trees with variance σ²:
- Variance of single tree: σ²
- Variance of average: σ²/B

In practice, trees are correlated (correlation ρ), so:
**Var(Random Forest) = ρσ² + ((1-ρ)/B)σ²**

This shows that:
- Increasing B always helps (reduces the second term)
- Reducing correlation ρ between trees helps (this is why we use feature randomness)

### Generalization Error
The generalization error of a random forest depends on:
1. **Strength of individual trees:** More accurate trees → better forest
2. **Correlation between trees:** Less correlation → better forest

**Goal:** Build strong individual trees that are as uncorrelated as possible.

---

## Practical Guidelines

1. **Number of Trees:** Start with 100-500; more is generally better but has diminishing returns
2. **max_features:** Use √m for classification, m/3 for regression as starting points
3. **Tree Depth:** Usually don't limit (let trees grow fully); random forest's averaging handles overfitting
4. **min_samples_leaf:** Can set to 1-5 to control smoothness of predictions
5. **Use OOB Error:** For hyperparameter tuning instead of cross-validation (faster)

---

## Conclusion

Random forests improve upon single decision trees by:
- Training multiple diverse trees through bootstrap sampling and feature randomness
- Averaging predictions to reduce variance
- Maintaining low bias through unpruned trees
- Providing robust, accurate predictions for both classification and regression tasks

The mathematical foundation shows that the key to success is balancing the strength of individual trees with their diversity, achieved through the clever combination of bagging and random feature selection.