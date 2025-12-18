# Decision Trees - Mathematical Foundation

## Overview
Every node in a decision tree is labelled by a question about an attribute from the given dataset. Traverse the tree in this way until a leaf node is reached which gives us our decision. Main issue is choosing the attribute that should be used at a given node, that is, the question to be asked at each node.

## Objective
Seek to minimize number of questions required to obtain an outcome, which corresponds to finding a tree of minimum height.

---

## Entropy
Let P denote a (discrete) probability distribution, P = {p₁, p₂, ..., pₙ} with 0 ≤ pᵢ ≤ 1 for each i and Σ(pᵢ) = 1

**The entropy of P is:**

H(P) = -Σ(pᵢ × log(pᵢ))

**Example:** P = {0.5, 0.5} → H(P) = -Σ(0.5 × log(0.5)) = 1

Entropy is a measure of the randomness, or uncertainty in a system. In the example above entropy is at its maximum because event 1 and event 2 are equally likely to occur.

---

## Gini Impurity - an alternative to entropy

Gini(P) = 1 - Σ(pᵢ)²

The minimum Gini Impurity occurs when all the datapoints have the same target and this is called a pure set. Max occurs at P = (1/k, 1/k, ..., 1/k) because as k is increased the Gini Impurity tends towards 1.

---

## Constructing a tree from a dataset
Consider a dataset S consisting of an array X of datapoints and an array T of corresponding target values. Let F₁, F₂, ..., Fₙ be attributes of the datapoints in X (features).

For each attribute F, define Gain(S,F) as follows:

**Gain(S,F) = H(S) - 1/|S| × Σ(|Sₓ| × H(Sₓ))**

where the sum is over all values f in the domain of F, and Sₓ is the subset of S where attribute F has value f.

To determine the attribute to choose at the root node, we calculate Gain(S,F) for each attribute F and choose the attribute for which Gain(S,F) is maximum.

*Note: H(S) and 1/|S| are unaffected by choice of F so only need to calculate the other part of the equation.*

---

## Example: Calculating Information Gain

Consider a dataset of 14 weather observations for deciding whether to play tennis:

| Day | Outlook  | Temperature | Play Tennis |
|-----|----------|-------------|-------------|
| 1   | Sunny    | Hot         | No          |
| 2   | Sunny    | Hot         | No          |
| 3   | Overcast | Hot         | Yes         |
| 4   | Rain     | Mild        | Yes         |
| 5   | Rain     | Cool        | Yes         |
| 6   | Rain     | Cool        | No          |
| 7   | Overcast | Cool        | Yes         |
| 8   | Sunny    | Mild        | No          |
| 9   | Sunny    | Cool        | Yes         |
| 10  | Rain     | Mild        | Yes         |
| 11  | Sunny    | Mild        | Yes         |
| 12  | Overcast | Mild        | Yes         |
| 13  | Overcast | Hot         | Yes         |
| 14  | Rain     | Mild        | No          |

**Dataset S:** 14 total examples, 9 "Yes", 5 "No"

### Step 1: Calculate H(S)

H(S) = -(9/14 × log₂(9/14)) - (5/14 × log₂(5/14))  
H(S) = -(9/14 × -0.637) - (5/14 × -1.485)  
H(S) = 0.410 + 0.531 = **0.940**

### Step 2: Calculate Gain(S, Outlook)

**Outlook values:** Sunny (5), Overcast (4), Rain (5)

- **S_Sunny:** 5 examples → 2 Yes, 3 No  
  H(S_Sunny) = -(2/5 × log₂(2/5)) - (3/5 × log₂(3/5)) = 0.971

- **S_Overcast:** 4 examples → 4 Yes, 0 No  
  H(S_Overcast) = -(4/4 × log₂(4/4)) - 0 = 0

- **S_Rain:** 5 examples → 3 Yes, 2 No  
  H(S_Rain) = -(3/5 × log₂(3/5)) - (2/5 × log₂(2/5)) = 0.971

**Gain(S, Outlook) = H(S) - 1/|S| × Σ(|Sₓ| × H(Sₓ))**  
Gain(S, Outlook) = 0.940 - 1/14 × [(5 × 0.971) + (4 × 0) + (5 × 0.971)]  
Gain(S, Outlook) = 0.940 - 1/14 × [4.855 + 0 + 4.855]  
Gain(S, Outlook) = 0.940 - 0.694 = **0.246**

### Step 3: Calculate Gain(S, Temperature)

**Temperature values:** Hot (4), Mild (6), Cool (4)

- **S_Hot:** 4 examples → 2 Yes, 2 No  
  H(S_Hot) = -(2/4 × log₂(2/4)) - (2/4 × log₂(2/4)) = 1.0

- **S_Mild:** 6 examples → 4 Yes, 2 No  
  H(S_Mild) = -(4/6 × log₂(4/6)) - (2/6 × log₂(2/6)) = 0.918

- **S_Cool:** 4 examples → 3 Yes, 1 No  
  H(S_Cool) = -(3/4 × log₂(3/4)) - (1/4 × log₂(1/4)) = 0.811

**Gain(S, Temperature) = H(S) - 1/|S| × Σ(|Sₓ| × H(Sₓ))**  
Gain(S, Temperature) = 0.940 - 1/14 × [(4 × 1.0) + (6 × 0.918) + (4 × 0.811)]  
Gain(S, Temperature) = 0.940 - 1/14 × [4.0 + 5.508 + 3.244]  
Gain(S, Temperature) = 0.940 - 0.911 = **0.029**

### Conclusion:
Since Gain(S, Outlook) = 0.246 > Gain(S, Temperature) = 0.029, we choose **Outlook** as the root node attribute.

---

## Decision Tree Construction Process

The process is then repeated on each child node (decision) of the root node excluding it from the process. A leaf node is created in one of two ways:

1. **If all datapoints in the current dataset have the same target,** then a leaf node is created with that target.

2. **If the current dataset has no more attributes left to test,** then the leaf node is created. In this case, it may happen that the targets are not the same. Then the label chosen for the leaf node is the most common target in the current dataset.

---

## ID3 Algorithm

Given a dataset with attributes F₁, ..., Fₘ which are all discrete and discrete classification values for target, a decision tree is built as follows:

- Initially, assign S to be the whole dataset
  - If all datapoints in S have the same target value, create a leaf node and label it with that target value.
  - Else if there are no attributes left to test, create a leaf node and label it with the target value that is most common in S.
  - Else find feature F* that maximizes the information gain:
    - For each value f in the domain of F*:
      - Add a new branch and node
      - Calculate Sₓ by selecting only those datapoints with F*-value equal to f
      - Remove attribute F*
      - Recursively call algorithm on dataset Sₓ

---

## Pruning a decision tree

To avoid overfitting, the method of pruning a tree can be used. Replace any subtree with a leaf node and give the node a label that is the most common target of all the datapoints that would be decided in the subtree.

Consider every data point in a dataset whose prediction would be obtained by following a path that ends up in a leaf within this subtree:

- Use validation set to decide where to prune a tree, init tree T.
  - Calculate error on validation set as follows:
    - **error(T) = (num of misclassifications) / (num of datapoints)**
  - Choose any subtree that you are considering pruning and let T' be the pruned tree. Calculate the validation set error on the pruned tree. If error(T') < error(T) then replace the tree T with the pruned tree T'; otherwise keep the original.
    - After pruning is complete, test data set is used to evaluate the tree.