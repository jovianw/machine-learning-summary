# Machine Learning Summary

This is a cheatsheet for me to understanding machine-learning techniques and methods, and it might help someone else out there too.

## Contents
- [Introduction](#machine-learning-summary)
    - [Variable Definitions](#variable-definitions)
- [Models](#models)
    - [Simple Regression](#simple-regression)
    - [Locally Weighted Regression](#locally-weighted-regression)
    - [Logistic Regression](#logistic-regression)
    - [Gaussian Discriminant Analysis](#gaussian-discriminant-analysis)
    - [Support Vector Machines (SVMs)](#support-vector-machines)
- [Techniques](#techniques)
    - [Cross Validation](#cross-validation)
    - [Regularization](#regularization)

## Variable Definitions

- $x$: input feature
- $X$: design matrix (first column is 1's)
- $y$: target, output variable
- $h(x)$: prediction of y
- $i$: training sample
- $m$: total number of training samples
- $n$: total number of features
- $\theta$: parameter
- $\tau$: variance
- $w$: weight
- $W$: weight matrix
- $\lambda$: regularization parameter

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import random
import sklearn.datasets 
import time
```

# Models

## Simple Regression

- Model continuous $y$ with a "good" line
- Regression, $y$ is continuous
- $h(x) = \theta_{0} + \theta_{1} + \dots = \bar{\theta}^{T}\bar{x} = X\theta$
- Cost function: $J(\theta)=\frac{1}{2m}=\displaystyle\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^{2}=\displaystyle\sum_{i=1}^{m}(\bar{\theta}^{T}\bar{x}^{(i)}-y^{(i)})^{2}$
- Closed form solution: $\bar{\theta}=(X^{T}X)^{-1}X^{T}\bar{y}$
    - No $\alpha$
    - Need to compute matrix inverse, slow
- Batch gradient descent: $\theta_{j} = \theta_{j}-\alpha\frac{1}{m}\displaystyle\sum_{i=1}^{m}(\bar{\theta}^{T}\bar{x}^{(i)}-y^{(i)})x_{j}^{(i)}$
    - Always converges to optimum
    - Costly for large datasets, slow
- Stochastic gradient descent: $\theta_{j} = \theta_{j}-\alpha\frac{1}{m}(\bar{\theta}^{T}\bar{x}^{(i)}-y^{(i)})x_{j}^{(i)}$, repeat for each training sample
    - Oscillates
    - Makes early progress, usually good enough

```python
class PolynomialRegressionModel():
    
    '''
    Implementation of a Polynomial regression model using MSE and Gradient Descent.
    @param X (numpy.ndarray) - training set of input features, as a design matrix
           y (numpy.ndarray) - training set of corresponding targets
           theta (numpy.ndarray) - model parameters (variable coefficients, in order of increasing variable degree)
           alpha (float) - learning step size
           degree (int) - highest polynomial degree
    '''
    def __init__(self, X, y, theta, alpha, degree):
        self.X = X
        self.y = y
        self.theta = theta
        self.alpha = alpha
        self.degree = degree
             
    '''
    Hypothesis - return model prediction 
    @param X (numpy.ndarray) - design matrix of input features
    @return corresponding model output (float)
    '''
    def h(self, X):
        return np.dot(X, self.theta)
    
    '''
    Renders a plot of the training data and the regression line based on current model parameters.
    ''' 
    def plot_current_model(self):
        reg_curve_y = self.h(self.X)
        plt.scatter(self.X[:,1], self.y)
        plt.scatter(self.X[:,1], reg_curve_y, color='green')
        plt.title('California Housing (scaled data)')
        plt.xlabel('Median Income')
        plt.ylabel('Median Price')
        plt.show()
    
    '''
    Cost function measuring mean squared error of the regression line for a given training set and model parameters.
    Vectorize your code - no loops!
    @return MSE based on the current parameters (float)
    '''
    def J(self):
        return np.sum(((self.h(self.X) - self.y) ** 2) / (2 * np.shape(self.X)[0]))
    
    '''
    Update theta for one gradient descent step. Vectorize your code - no loops!
    @return the gradient of the cost function (numpy.ndarray), for use in run_gradient_descent
    '''
    def gradient_descent_step(self):
        gradient = np.dot(self.X.T, (np.dot(self.X, self.theta) - self.y)) / np.shape(self.X)[0]
        self.theta = self.theta - (self.alpha * gradient)
        return gradient
       
    '''
    Run gradient descent to optimize the model parameters.
    Keep track of the losses. You may change the default threshold for convergence. 
    Here, we will use a convergence criterion based on the norm of the gradient vector.
    @param threshold (float) - run gradient descent until the absolute norm of the gradient is below this value.
    @return a list storing the value of the cost function after every step of gradient descent (float list)
    '''
    def run_gradient_descent(self, threshold=0.01):
        losses = []
        loss = self.J()
        losses.append(loss)
        
        norm_grad_vec = 1
        while norm_grad_vec > threshold:
            grad_vec = self.gradient_descent_step()
            loss = self.J()
            losses.append(loss)
            norm_grad_vec = np.linalg.norm(grad_vec)
        return losses
    
    '''
    Renders the learning curve of the model during its optmization process. 
    @param losses (float list) - MSE after every gradient descent step 
    '''
    def plot_MSE_loss(self, losses):
        plt.plot(range(len(losses)), losses)
        plt.title('Learning Curve')
        plt.xlabel('Number of Steps')
        plt.ylabel('MSE Loss')
        plt.show()
```

---

## Locally Weighted Regression

- Fit model using data near the query point
- Use weight function $w^{(i)}=e^{-\frac{(x^{(i)}-\bar{x})^{2}}{2\tau^{2}}}$
- Cost function: $J(\theta)=(X\theta-\bar{y})^{T}W(X\theta-\bar{y})$
- Closed form solution: $\theta=(X^{T}WX)^{-1}X^{T}W\bar{y}$
- Requires $\tau$
- Need large, densely sampled data, computationally intensive

```python
class LocallyWeightedLR():
    
    def __init__(self, X, y, tau):
        self.X = X
        self.y = y
        self.tau = tau 
        
    # use bandwidth variable tau to compute weights for each training point.  
    # return a diagonal matrix with w_i on the diagonal (for vectorization)
    # note that the values of w_i depend upon the value of the input query point x.
    def compute_weights(self, x):
        xi = self.X[:, 1]
        w = np.exp(np.square(xi - x) / (-2 * (self.tau ** 2)))
        W = np.diag(w)
        return W
    
    # analytical solution for the local linear regression parameters at the input query point x.
    # this should involve calling the above method compute_weights.
    def compute_theta(self, x):
        W = self.compute_weights(x)
        theta = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(self.X.T, W), self.X)), self.X.T), W), y)
        return theta
    
    # prediction for an input x
    # also return the local linear regression parameters (theta) for this x.
    def predict(self, x):
        theta = self.compute_theta(x)
        prediction = theta[0, 0] + (theta[1, 0] * x)
        return (prediction, theta)
```

---

## Logistic Regression

- Estimate the probability that $y=1$ as opposed to $y=0$
- Classification, $y$ is discrete
- $\theta$ estimates the boundary line
- $h(x)=\frac{1}{1+e^{-\theta^{T}x}}$
- Decision boundary: $h_{\theta}(x)=0.5$ or $\theta^{T}x=0$
- Cost function: $J(\theta)=-\frac{1}{m}\displaystyle\sum_{i=1}^{m}y^{(i)}\log(h_{\theta}(x^{(i)}))+(1-y^{(i)})\log(1-h_{\theta}(x^{(i)}))$
- No closed form solution
- Gradient descent: $\theta_{j}=\theta_{j}-\alpha\displaystyle\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}$

```python
class LogisticRegression():
    
    # X is design matrix, y is mx1, theta is mx1, alpha is numeric
    def __init__(self, X, y, theta, alpha):
        self.X = X
        self.y = y
        self.theta = theta 
        self.alpha = alpha
        self.losses = []
    
    #  h (hypothesis): returns p(y=1|x) on inputs contained in the design matrix X
    def sigmoid(self, X): 
        return (1 / (1 + np.exp(-np.dot(X, self.theta))))
    
    # return predictions of class membership (0,1) of the datapoints in an input matrix X
    def predict(self, X):
        h = self.sigmoid(X)
        h[h >= 0.5] = 1 # Convert by h value
        h[h < 0.5] = 0
        return h
    
    # cost function J()
    def cost(self):
        h = self.sigmoid(self.X)
        m = np.shape(self.X)[0]
        if np.any(h == 0) or np.any(h == 1): # Occurs when precision isn't high enough, just assume infinite cost
            return float('inf')
        result = np.sum(((self.y * np.log(h)) + ((1 - self.y) * np.log(1 - h))) / (-m))
        return result
    
    # update theta 
    def gradient_descent_step(self):
        gradient = np.dot(self.X.T, (self.sigmoid(X) - self.y))
        self.theta = self.theta - (self.alpha * gradient)
        return gradient
    
    # define a convergence criterion 
    # run gradient descent until convergence 
    def run_gradient_descent(self, threshold=0.01):
        self.losses = [self.cost()]
        
        norm_grad_vec = float('inf')
        while norm_grad_vec > threshold:
            grad_vec = self.gradient_descent_step()
            loss = self.cost()
            self.losses.append(loss)
            norm_grad_vec = np.linalg.norm(grad_vec)
    
    # return the model's accuracy on an input (X,y) dataset 
    def evaluate(self, X, y):
        return 1 - (np.sum(np.abs(self.predict(X) - y)) / np.shape(X)[0])
    
    # plot cost function over num gradient descent steps
    def learning_curve(self):
        plt.plot(range(len(self.losses)), self.losses)
        plt.title('Learning Curve')
        plt.xlabel('Number of Steps')
        plt.ylabel('Cost')
        plt.show()
    
    # plot decision boundary, based on current model parameters
    # you may edit or add cases to this, to accommodate plotting the Iris data too
    def decision_boundary(self, dset):
        X = self.X[:,1:]
        theta = [t[0] for t in self.theta]
        y = np.reshape(self.y, (-1))
        xax = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
        yax = (theta[0] + (theta[1] * xax)) / (-theta[2])
        plt.plot(xax, yax) # Boundary line
        plt.scatter(x=X[y==0,0],y=X[y==0,1],c='red',edgecolor='black')
        plt.scatter(x=X[y==1,0],y=X[y==1,1],c='blue',edgecolor='black')
        plt.show()
```

---

## Gaussian Discriminant Analysis

- Try to estimate $p(x|y)$ and $p(y)$ for each label $y$
- Generative algorithm, classification

---

## Support Vector Machines

- 

# Techniques

## Cross Validation

- 

## Regularization

- Add a weight to unimportant $\theta$ in the cost function to encourage small $\theta$
- For linear or logistic regression
- Select $\lambda$ using cross-validation
- Cost function: $J(\theta)=\dots\frac{1}{m}\left((\text{measure of fit})+\lambda\displaystyle\sum_{j=1}^{n}\theta_{j}^{2}\right)$ except for $\theta_{0}$
- Closed form solution for linear regression: 
$$\theta=\left(X^{T}X+\lambda
\begin{bmatrix}
0\\
& 1\\
& & 1\\
& & & \ddots\\
& & & & 1\\
\end{bmatrix}
\right)^{-1}X^{T}\bar{y}$$
- Gradient descent: $\theta_{j}=\theta_{j}-\alpha(\frac{1}{m}\left((\text{derivative of measure of fit})+\lambda\theta_{j}\right))$ except for $\theta_{0}$