# Ridge Regression - Mathematical Foundation

## Objective
Find the line of best fit using a regression approach by minimizing prediction error **with L2 regularization** to prevent overfitting.

## Model
```
y = mx + b
```
Where:
- `y` = predicted output
- `m` = slope (weight)
- `x` = input feature
- `b` = y-intercept (bias)

## Regularization Concept

**Problem with standard Linear Regression:**
- Can overfit when features are highly correlated (multicollinearity)
- Large weights can lead to high variance
- Model becomes too complex and sensitive to training data

**Ridge Solution:**
- Add penalty term that discourages large weights
- Forces the model to be simpler and more generalizable
- Shrinks coefficients toward zero (but never exactly zero)

## Loss Function: Mean Squared Error + L2 Penalty

```
E = (1/n) × Σ(yᵢ - (mxᵢ + b))² + λ × Σ(mⱼ²)
```

Breaking this down:
- **First term**: Standard MSE (measures prediction error)
- **Second term**: L2 regularization penalty (sum of squared weights)
- **λ (lambda)**: Regularization strength (hyperparameter)

**Key Points:**
- When λ = 0: Ridge reduces to standard linear regression
- When λ → ∞: All weights shrink to 0
- Bias term `b` is **not** penalized (only weights are regularized)
- L2 penalty = `λ × ||m||²` where ||m||² is the squared L2 norm

**Why L2 (Ridge) over L1 (Lasso)?**
- L2: Shrinks all coefficients proportionally, keeps all features
- L1: Can shrink coefficients to exactly zero, performs feature selection
- L2: Has a unique solution, always differentiable
- L2: Better when all features are potentially relevant

## Gradient Descent

To minimize the error, we calculate partial derivatives with respect to `m` and `b`, including the regularization term.

### Partial Derivative w.r.t. m (slope)

```
∂E/∂m = ∂/∂m[(1/n) × Σ(yᵢ - (mxᵢ + b))²] + ∂/∂m[λ × Σ(mⱼ²)]
      = (1/n) × Σ[2(yᵢ - (mxᵢ + b)) × (-xᵢ)] + 2λm
      = -(2/n) × Σ[xᵢ(yᵢ - (mxᵢ + b))] + 2λm
```

**Important**: The regularization term adds `2λm` to the gradient.

### Partial Derivative w.r.t. b (bias)

```
∂E/∂b = (1/n) × Σ[2(yᵢ - (mxᵢ + b)) × (-1)]
      = -(2/n) × Σ(yᵢ - (mxᵢ + b))
```

**Note**: Bias is **not regularized**, so this is identical to standard linear regression.

## Update Rules

Move in the opposite direction of the gradient to minimize error:

```
m = m - α × (∂E/∂m)
  = m - α × [-(2/n) × Σ[xᵢ(yᵢ - (mxᵢ + b))] + 2λm]
  = m - α × [-(2/n) × Σ[xᵢ(yᵢ - (mxᵢ + b))]] - α × 2λm
  = m × (1 - 2αλ) - α × [-(2/n) × Σ[xᵢ(yᵢ - (mxᵢ + b))]]

b = b - α × (∂E/∂b)
  = b - α × [-(2/n) × Σ(yᵢ - (mxᵢ + b))]
```

**Key Insight**: The weight update includes a **shrinkage factor** `(1 - 2αλ)` that pulls weights toward zero each iteration.

Where:
- `α` = learning rate (typically 0.01)
- `λ` = regularization strength (typically 0.01 to 10)
- Controls the step size and amount of regularization

## Matrix Form (for multiple features)

For vectorized implementation with multiple features:

```
Loss: E = (1/n)||y - Xm - b||² + λ||m||²

Gradient w.r.t. m: ∇ₘE = -(2/n)Xᵀ(y - Xm - b) + 2λm
Gradient w.r.t. b: ∇ᵦE = -(2/n)Σ(y - Xm - b)

Update: m = m(1 - 2αλ) - α∇ₘE
        b = b - α∇ᵦE
```

## Implementation Notes

1. **Initialization**: Start with `m = 0` and `b = 0` (or small random values)
2. **Feature Scaling**: **Critical** for Ridge - features must be normalized/standardized
   - Ridge penalizes larger-magnitude features more heavily
   - Without scaling, features on different scales are penalized differently
3. **Choosing λ**:
   - Use cross-validation to find optimal λ
   - Start with λ ∈ [0.01, 0.1, 1, 10, 100]
   - Larger λ = more regularization = simpler model
4. **Convergence**: Stop when loss change is below threshold or max iterations reached
5. **Learning Rate**: Balance between α and λ - both affect weight magnitude