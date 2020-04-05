import numpy as np
from FOSModel import FOSModel

mu, sigma = 0, 3.5 # mean and standard deviation
x = np.random.normal(mu, sigma, 1000)
y = np.sin(x)

# where max_m is the max number of  terms, mse_reduction_threshold is the early-stop criteria
model = FOSModel(max_delay_in_input=4, max_delay_in_output=0, max_order=2, max_m=10, mse_reduction_threshold=1e-4)

# x (input) and y (output) are 1-dimentional series
model.fit(x, y)

# all x points and a few initial points of y, predicted y series will be returned
y1 = model.predict(x, y)

mse = np.mean((y1 - y)**2)
