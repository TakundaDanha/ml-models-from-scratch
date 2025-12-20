# Regression Tree - Mathematical Modelling

## Overview
A decision tree can also be constructed for a regression problem, that is, a dataset in which target variables are real values. For such a problem Entropy and Gini are not appropriate since they measure categorical uncertainty. Instead, we use Mean Squared Error (MSE) to measure the variance in continuous target values.

## Objective
Seek to minimize the prediction error across the dataset, which corresponds to finding splits that create homogeneous subsets with respect to the target values.

---

## Mean Squared Error
If the column of target values for dataset S is [t₁, t₂, ..., tₙ], then the mean squared error of S, denoted MSE(S), is obtained by first calculating the mean of the target variables:

**μ(S) = (1/n) × Σ(tᵢ)**

and then computing:

**MSE(S) = (1/n) × Σ([tᵢ - μ(S)]²)**

If all target values in the dataset are close, then MSE is small. The more spread out the target values, the greater the MSE becomes. If all values are identical, then MSE = 0.

**Example:** For dataset S with target values [10, 12, 14]:
- μ(S) = (10 + 12 + 14)/3 = 12
- MSE(S) = [(10-12)² + (12-12)² + (14-12)²]/3 = [4 + 0 + 4]/3 = 2.67

---

## Constructing a tree from a dataset
Consider a dataset S consisting of an array X of datapoints and an array T of corresponding target values. Let F₁, F₂, ..., Fₘ be attributes of the datapoints in X (features).

For each attribute F, define Gain(S,F) as follows:

**Gain(S,F) = MSE(S) - 1/|S| × Σ(|Sբ| × MSE(Sբ))**

where the sum is over all values f in the domain of F, and Sբ is the subset of S where attribute F has value f.

To determine the attribute to choose at the root node, we calculate Gain(S,F) for each attribute F and choose the attribute for which Gain(S,F) is maximum. This is equivalent to choosing the split that produces the greatest reduction in MSE.

*Note: MSE(S) and 1/|S| are unaffected by choice of F so only need to calculate the weighted average MSE part of the equation.*

---

## Example: Predicting House Prices

Consider a dataset of 12 houses for predicting price (in thousands):

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

**Dataset S:** 12 total examples with target prices

### Step 1: Calculate MSE(S)

First, calculate the mean price:
μ(S) = (150 + 160 + 200 + 210 + 280 + 140 + 145 + 190 + 250 + 170 + 220 + 230)/12  
μ(S) = 2345/12 = **195.42**

Now calculate MSE(S):
MSE(S) = [(150-195.42)² + (160-195.42)² + (200-195.42)² + (210-195.42)² + (280-195.42)² + (140-195.42)² + (145-195.42)² + (190-195.42)² + (250-195.42)² + (170-195.42)² + (220-195.42)² + (230-195.42)²]/12

MSE(S) = [2062.47 + 1254.47 + 20.97 + 212.47 + 7153.67 + 3071.47 + 2542.47 + 29.37 + 2977.67 + 646.47 + 604.67 + 1195.67]/12  
MSE(S) = 21771.84/12 = **1814.32**

### Step 2: Calculate Gain(S, Location)

**Location values:** Urban (5), Suburban (4), Rural (3)

- **S_Urban:** Houses 1-5 → Prices: [150, 160, 200, 210, 280]  
  μ(S_Urban) = 1000/5 = 200  
  MSE(S_Urban) = [(150-200)² + (160-200)² + (200-200)² + (210-200)² + (280-200)²]/5  
  MSE(S_Urban) = [2500 + 1600 + 0 + 100 + 6400]/5 = **2120**

- **S_Suburban:** Houses 6-9 → Prices: [140, 145, 190, 250]  
  μ(S_Suburban) = 725/4 = 181.25  
  MSE(S_Suburban) = [(140-181.25)² + (145-181.25)² + (190-181.25)² + (250-181.25)²]/4  
  MSE(S_Suburban) = [1700.56 + 1314.06 + 76.56 + 4726.56]/4 = **1954.44**

- **S_Rural:** Houses 10-12 → Prices: [170, 220, 230]  
  μ(S_Rural) = 620/3 = 206.67  
  MSE(S_Rural) = [(170-206.67)² + (220-206.67)² + (230-206.67)²]/3  
  MSE(S_Rural) = [1344.89 + 177.69 + 544.89]/3 = **689.16**

**Gain(S, Location) = MSE(S) - 1/|S| × Σ(|Sբ| × MSE(Sբ))**  
Gain(S, Location) = 1814.32 - 1/12 × [(5 × 2120) + (4 × 1954.44) + (3 × 689.16)]  
Gain(S, Location) = 1814.32 - 1/12 × [10600 + 7817.76 + 2067.48]  
Gain(S, Location) = 1814.32 - 1/12 × 20485.24  
Gain(S, Location) = 1814.32 - 1707.10 = **107.22**

### Step 3: Calculate Gain(S, Size)

**Size values:** Small (4), Medium (4), Large (4)

- **S_Small:** Houses 1, 2, 6, 7 → Prices: [150, 160, 140, 145]  
  μ(S_Small) = 595/4 = 148.75  
  MSE(S_Small) = [(150-148.75)² + (160-148.75)² + (140-148.75)² + (145-148.75)²]/4  
  MSE(S_Small) = [1.56 + 126.56 + 76.56 + 14.06]/4 = **54.69**

- **S_Medium:** Houses 3, 4, 8, 10 → Prices: [200, 210, 190, 170]  
  μ(S_Medium) = 770/4 = 192.5  
  MSE(S_Medium) = [(200-192.5)² + (210-192.5)² + (190-192.5)² + (170-192.5)²]/4  
  MSE(S_Medium) = [56.25 + 306.25 + 6.25 + 506.25]/4 = **218.75**

- **S_Large:** Houses 5, 9, 11, 12 → Prices: [280, 250, 220, 230]  
  μ(S_Large) = 980/4 = 245  
  MSE(S_Large) = [(280-245)² + (250-245)² + (220-245)² + (230-245)²]/4  
  MSE(S_Large) = [1225 + 25 + 625 + 225]/4 = **525**

**Gain(S, Size) = MSE(S) - 1/|S| × Σ(|Sբ| × MSE(Sբ))**  
Gain(S, Size) = 1814.32 - 1/12 × [(4 × 54.69) + (4 × 218.75) + (4 × 525)]  
Gain(S, Size) = 1814.32 - 1/12 × [218.76 + 875 + 2100]  
Gain(S, Size) = 1814.32 - 1/12 × 3193.76  
Gain(S, Size) = 1814.32 - 266.15 = **1548.17**

### Conclusion:
Since Gain(S, Size) = 1548.17 > Gain(S, Location) = 107.22, we choose **Size** as the root node attribute because it provides the greatest reduction in MSE.

---

## Regression Tree Construction Process

The process is then repeated on each child node of the root node, excluding the selected attribute from further consideration. A leaf node is created in one of two ways:

1. **If all datapoints in the current dataset have sufficiently similar target values** (within a threshold), then a leaf node is created with the mean of those target values as the prediction.

2. **If the current dataset has no more attributes left to test,** or if further splitting doesn't significantly reduce MSE, then a leaf node is created with the mean of the target values in that subset as the prediction.

---

## Prediction with Regression Trees

Once the tree is built, prediction for a new datapoint is made by:
1. Traversing the tree based on the datapoint's feature values
2. Reaching a leaf node
3. Returning the mean target value stored at that leaf node

In our example, a house with Size = "Large" would be predicted to have a price of 245 (the mean of the Large subset).

---

## Pruning a Regression Tree

To avoid overfitting, pruning can be used. Replace any subtree with a leaf node containing the mean of all target values that would be predicted in that subtree.

Use a validation set to decide where to prune:
- Calculate error on validation set:
  - **error(T) = MSE on validation set**
- Choose any subtree for pruning and let T' be the pruned tree
- Calculate validation set error on pruned tree
- If error(T') < error(T), replace T with T'; otherwise keep original
- After pruning is complete, use the test dataset to evaluate final performance

---

## Properties of Regression Trees

**Advantages:**
- Invariant under scaling and monotonic transformations of feature values
- Robust to inclusion of irrelevant features
- Produce models that are interpretable and explainable
- Handle non-linear relationships naturally

**Disadvantages:**
- Prone to overfitting on training sets, especially with deeper trees
- Learn irregular, discontinuous patterns (predictions change sharply at split boundaries)
- Have low bias but very high variance
- Small changes in data can lead to very different tree structures