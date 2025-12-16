# Linear Regression - Mathematical Foundation

## Objective
Find the line of best fit using a regression approach by minimizing prediction error.

## Model
```
y = mx + b
```
Where:
- `y` = predicted output
- `m` = slope (weight)
- `x` = input feature
- `b` = y-intercept (bias)

## Loss Function: Mean Squared Error (MSE)

```
E = (1/n) × Σ(yᵢ - (mxᵢ + b))²
```

**Why MSE over MAE?**
- MSE penalizes larger errors more heavily (squared term)
- MAE treats all errors equally
- MSE provides smoother gradients for optimization

## Gradient Descent

To minimize the error, we calculate partial derivatives with respect to `m` and `b`, then move in the opposite direction of the gradient.

### Partial Derivative w.r.t. m (slope)

```
∂E/∂m = (1/n) × Σ[2(yᵢ - (mxᵢ + b)) × (-xᵢ)]
      = -(2/n) × Σ[xᵢ(yᵢ - (mxᵢ + b))]
```

### Partial Derivative w.r.t. b (bias)

```
∂E/∂b = (1/n) × Σ[2(yᵢ - (mxᵢ + b)) × (-1)]
      = -(2/n) × Σ(yᵢ - (mxᵢ + b))
```

## Update Rules

These gradients give us the direction of steepest ascent. We move in the **opposite direction** to minimize error:

```
m = m - α × (∂E/∂m)
b = b - α × (∂E/∂b)
```

Where:
- `α` = learning rate (typically 0.01)
- Controls the step size for each update

## Implementation Notes

1. **Initialization**: Start with `m = 0` and `b = 0` (or small random values)
2. **Iteration**: Repeat update rules until convergence
3. **Convergence**: Stop when error change is below threshold or max iterations reached
4. **Learning Rate**: Too high → overshooting, too low → slow convergence