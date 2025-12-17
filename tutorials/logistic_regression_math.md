# Logistic Regression - Mathematical Foundation

## Objective:
Used for classification, Logistic Regression outputs a probability that indicates how likely the output falls into a class. Does not give 0 or 1 but a probability, this is why its called regression becuase it predicts a value on a continuous scale and then we use that probability guided by a threshold to determine class. We do an ordinary linear regression then applying a function that forces the values between 0 and 1.

n features(n parameters) and m values

# Logistic Regression - Mathematical Foundation

## Objective
Classify binary outcomes (0 or 1) by estimating the probability that an instance belongs to a particular class.

## Model
```
θᵀx⁽ⁱ⁾ + b = z⁽ⁱ⁾, where z⁽ⁱ⁾ is the logits
```
Where:
- `z⁽ⁱ⁾` = predicted output
- `θᵀ (shortened to t)` = parameters (weight)
- `x⁽ⁱ⁾` = input feature
- `b` = y-intercept (bias)

## Sigmoid Function
Forces the output to be between 0 and 1
```
σ(z) = 1/(1+e⁻ᶻ)
```

## Estimated Probability
```
h_θ(x⁽ⁱ⁾) = σ(z⁽ⁱ⁾)
```

## Loss Function

**Likelihood:**
```
l(θ) = Π(i=1 to m) [h_θ(x⁽ⁱ⁾)]^y⁽ⁱ⁾ [1 - h_θ(x⁽ⁱ⁾)]^(1-y⁽ⁱ⁾)
```

1. Negate it to do gradient descent - minimize
2. Numerically unstable - take the natural logarithm
3. Want to make it scale invariant - divide by m

**Log Likelihood:**
```
L = ln(l(θ)) = Σ(y⁽ⁱ⁾ ln(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) ln(1 - h_θ(x⁽ⁱ⁾)))
```

**Cross-Entropy Loss:**
```
J(θ) = -(1/m) Σ(y⁽ⁱ⁾ ln(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) ln(1 - h_θ(x⁽ⁱ⁾)))
```

In basic terms this is the negative average log likelihood is the cross entropy loss and we will use this to perform gradient descent.

## Gradient Descent - Partial Derivatives

```
J = (-1/m) × ln(l(θ))
∂J/∂L = -1/m
```

**Chain Rule:**
```
∂L/∂θⱼ = ∂L/∂h × ∂h/∂z × ∂z/∂θⱼ
```

```
∂L/∂h = y⁽ⁱ⁾ × 1/h + (1-y) × (-1/(1-h))
      = y/h - (1-y)/(1-h)
```

```
∂h/∂z = sigmoid(z)(1-sigmoid(z))
      = h(1-h)
```

```
∂z/∂θⱼ = xⱼ
```

```
∂L/∂θⱼ = [y/h - (1-y)/(1-h)][h(1-h)][xⱼ]
       = (y-h)xⱼ
```

```
∂J/∂θⱼ = (-1/m)[(y-h)xⱼ]
```

**Matrix Form:**
```
∇_θ J(θ) = (-1/m) Xᵀ[h(θX) - y]
         = (1/m) Xᵀ[y - h(θX)]
```

## Update Rules

These gradients give us the direction of steepest ascent. We move in the **opposite direction** to minimize error:

```
θ = θ - α × ∇_θ J(θ)
```

Where:
- `α` = learning rate (typically 0.01)
- Controls the step size for each update

## Implementation Notes

1. **Initialization**: Start with `θ = 0` (or small random values)
2. **Iteration**: Repeat update rules until convergence
3. **Convergence**: Stop when loss change is below threshold or max iterations reached
4. **Learning Rate**: Too high → oscillation, too low → slow convergence
