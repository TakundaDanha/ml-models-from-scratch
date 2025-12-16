# Try to find a line of best fit - regression approach - minimising error (MAE or MSE)
# y = mx + b

# Mean Squared Error: 
# E = 1/n(sum(yi-(mxi+b))**2)
# Means that greater errors are punished more harshly as opposed to mean absolute error where all errors are punished the same

# Partial derivatives with respect to m and b and go opposite direction to this gradient

# dE/dm = 1/n(sum(2(yi-(mxi+b))*(-xi)))
#       = -2/n(sum(xi(yi-(mxi+b))))

# dE/db = -2/n(sum(yi-(mxi+b)))

# This gives us the direction of the steepest ascent w.r.t m and b, now go opposite direction

# m = m - alpha(dE/dm)
# b = b - lpha(dE/db)

# Learning rate (alpha) is 0.01



 