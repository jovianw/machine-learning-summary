# Summary of the Fundamentals of Machine Learning

## Contents
- [Introduction](#machine-learning-summary)
    - [Variable Definitions](#variable-definitions)
- [Models](#models)
    - [Simple Regression](#simple-regression)
    - [Locally Weighted Regression](#locally-weighted-regression)
    - [Logistic Regression](#logistic-regression)
    - [Gaussian Discriminant Analysis](#gaussian-discriminant-analysis)
    - [Support Vector Machines (SVMs)](#support-vector-machines)
    - [Naive Bayes](#naive-bayes)
    - [K-Means Clustering](#k-means-clustering)
    - [Simple Neural Networks](#simple-neural-networks)
- [Techniques](#techniques)
    - [Cross Validation](#cross-validation)
    - [Feature Selection](#feature-selection)
    - [Principal Components Analysis](#principal-components-analysis-pca)
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
- $\Phi$: bernoulli distribution probability
- $\Sigma$: Sum or covariance matrix
- $l$: hidden layer
- $a^{[l]}$: input to hidden layer

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
- $p(y=\text{label}|\vec{x})=\frac{p(\vec{x}|y=\text{label}) p(y=\text{label})}{p(\vec{x})}$
- $p(\vec{x}|y=\text{label})=N(\vec{\mu_\text{label}}, \Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\left(-\frac{1}{2}(\vec{x}-\mu_\text{label})^T\Sigma^{-1}(\vec{x}-\mu_\text{label})\right)$
- $p(y=\text{label})=\phi_\text{label}$
- Maximum likelihood parameter estimations:  
    - $\Phi=\frac{\text{number of entries where y=label}}{\text{total number of entries}}$
    - $\mu_{\text{label}}^*=\text{average }\vec{x}\text{ when y=label}$
    - $\Sigma^*=\frac{1}{m}\displaystyle\sum_{i=1}^{m}(\vec{x}-\mu_{y^{(i)}})(\vec{x}-\mu_{y^{(i)}})^T$
- Decision boundary is linear if distribution $\sigma$ is shared, otherwise boundaries are quadratic
- Makes stronger assumptions on data; if assumptions are valid, will find better fit with less training.

```python
class GDA():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.mu0 = np.zeros((self.X.shape[1], 1))
        self.mu1 = np.zeros((self.X.shape[1], 1))
        self.sigma = np.zeros((X.shape[1], X.shape[1]))
        self.phi = 0
        
    def fit(self):
        m = len(self.y)
        m1 = self.y.sum()
        m0 = m - m1
        # Phi
        self.phi = m1 / m
        # M0
        self.mu0 = np.zeros((self.X.shape[1], 1))
        for i in range(len(self.y)):
            if self.y[i] == 0: self.mu0 += self.X[i:i+1,:].T
        self.mu0 /= m0
        # M1
        self.mu1 = np.zeros((self.X.shape[1], 1))
        for i in range(len(self.y)):
            if self.y[i] == 1: self.mu1 += self.X[i:i+1,:].T
        self.mu1 /= m1
        # Sigma
        self.sigma = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(len(self.y)):
            if self.y[i] == 0:
                self.sigma += np.dot((self.X[i:i+1,:].T - self.mu0), (self.X[i:i+1,:] - self.mu0.T))
            else:
                self.sigma += np.dot((self.X[i:i+1,:].T - self.mu1), (self.X[i:i+1,:] - self.mu1.T))
        self.sigma /= m
        return (self.phi, self.mu0, self.mu1, self.sigma)
    
    def plot(self):
        # Scatterplot
        plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='coolwarm')
        # Contour maps
        x, y = np.mgrid[self.X[:,0].min():self.X[:,0].max():0.1, self.X[:,1].min():self.X[:,1].max():0.1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y
        rv0 = sp.stats.multivariate_normal(self.mu0.ravel(), self.sigma)
        rv1 = sp.stats.multivariate_normal(self.mu1.ravel(), self.sigma)
        plt.contour(x, y, rv0.pdf(pos), levels=5, colors='blue', alpha=0.5)
        plt.contour(x, y, rv1.pdf(pos), levels=5, colors='red', alpha=0.5)
        plt.xlabel('Citric Acid')
        plt.ylabel('Total Sulfur Dioxide')
        plt.title('Gaussian Distributions')
        plt.show()
        
    def predict(self, testX):
        # flatten the training data
        testX = testX.reshape(testX.shape[0], -1)
        scores = np.zeros((testX.shape[0], 2))
        # probability of 0
        distribution0 = sp.stats.multivariate_normal(mean=self.mu0.ravel(), cov=self.sigma)
        for i in range(testX.shape[0]):
            scores[i, 0] = np.log(1 - self.phi) + distribution0.logpdf(testX[i,:])
        # probability of 1
        distribution1 = sp.stats.multivariate_normal(mean=self.mu1.ravel(), cov=self.sigma)
        for i in range(testX.shape[0]):
            scores[i, 1] = np.log(self.phi) + distribution1.logpdf(testX[i,:])
        predictions = np.argmax(scores, axis=1)
        return predictions

    def accuracy(self, test):
        testX = test[:, :-1]
        testY = test[:, -1:]
        return np.sum(self.predict(testX) == testY.ravel()) / test.shape[0]
```

---

## Support Vector Machines

- Binary Classification
- $y\in\{-1,1\}$
- Intuition: create a line that has the largest gap between the closest points of each class. The closest points are support vectors.
- $\displaystyle{\min_{w,b}\frac{1}{2}||w||^2}$, subject to $y^{(i)}(w^Tx^{(i)}+b)\ge1$, for $i=1\dots m$
- The process of solving this can be entirely written using inner products. Inner products can be replaced with some feature mapping $\Phi(.)$ to learn high dimensional feature spaces.
    - Polynomial kernels
    - Radial basis function kernel
- Soft-margin SVM optimization:  
$\displaystyle{\min_{\Phi,w,b}\frac{1}{2}||w||^2+C\displaystyle\sum_{i=1}^{m}\xi_i}$ such that $y^{(i)}(w^Tx^{(i)}+b)\ge 1-\xi_i$, for $i=1\dots m$ and $\xi\ge0$ (the slack variable)

---

## Naive Bayes

- Assumption: Given a training set $\{x^{(i)},y^{(i)}\}$, $x_i$ values are conditionally independent given y. ($x_i$ is binary variable)
- Using the chain rule of probability, $p(x_1,x_2,\dots,x_n|y)=p(x_1|y)p(x_2|y,x_1)\dots p(x_n|y,x_1,x_2,\dots,x_{n-1})$
- Combination: $p(x_1,x_2,\dots,x_n|y) = \displaystyle \prod_{j=1}^{n}p(x_j|y)$
- Parameters:   
$\Phi_{j|y=\text{label}}=p(x_j=1|y=\text{label})$  
$\Phi_y=p(y^{(i)})$
- Maximum Likelihood Estimations:  
$\Phi_{j|y=\text{label}}=$ fraction of entries with $y=\text{label}$ where $x_j=1$  
$\Phi_y=$ fraction of entries where $y=1$
- Output: $p(y=1|x)=\frac{p(x|y=1)p(y=1)}{p(x)}$  
- If training set is missing data and both $p(x|y=0)=0, p(x|y=1)=0$, apply Laplance smoothing  
$\Phi_{j|y=\text{label}}=\frac{(\text{number of entries where }x_j=1\text{ and }y=\text{label})+1}{(\text{number of entries where }y=\text{label})+2}$ 

---

## K-Means Clustering

- Unsupervised, no $y$ labels
- $k$ centroids: $\mu_1,\mu_2,\dots,\mu_k\in\mathbb{R}^{n}$
- Repeat until convergence:
    - Assign nearest centroid for every x: $\forall i, c^{(i)}=\displaystyle\mathop{\operatorname{arg\,min}}\limits_j||x^{(i)}-\mu_j||^2$
    - Move each centroid to the mean of assigned points: $\forall j, \mu_i=\frac{\displaystyle\sum_{i=1}^{m}1(c^{(i)}=j)x^{(i)}}{\displaystyle\sum_{i=1}^{m}1(c^{(i)}=j)}$
- Usually converges but could oscillate, could have mutliple random runs of initial $\mu_j$ 

---

## Simple Neural Networks

- Input features &rarr; Inputs layer &rarr; Hidden layer &rarr; Output layer &rarr; Output y
- Each input i to a hidden layer l is denoted activation $a_i^{[l]}$
- Each hidden unit includes inputs, some function $z=w^Tx+b$, some activation function $g(z)$, and output $a_i^{[l]}$
- Example activation functions:
    - Sigmoid/logistic: $\frac{1}{1+e^{-z}}$
    - ReLU: $\max(z, 0)$
    - tanh: $\frac{e^z-e^{-z}}{e^z+e^{-z}}$
    - Usually non-linear
- Vectorization
    - Can get $w^T$ from $W_i^{[l]T}$ where $W_i^{[l]T}$ is the ith row of the overall matrix of parameters for layer l, $W^{[l]}$
    - Similar process for z and x
    - $Z^{[l]}=W^{[l]}X+b^{[l]}$
- Initialize W and b to small random numbers (commonly $N(0, 0.1)$). Do not set to zero, otherwise output will always be 0.
- Backpropogation:
    - Consider a run of a nn using sigmoid activation function
    - Loss $L=-((1-y)\log(1-\hat y) + y\log \hat y)$
    - $W^{[l]}=W^{[l]}-\alpha \frac{dL}{dW^{[l]}}$
    - $b^{[l]}=b^{[l]}-\alpha \frac{dL}{db^{[l]}}$
    - After a lot of math, $\frac{dL}{dW^{[3]}}=(a^{[3]}-y)a^{[2]T}$
    - Chain rule:
        - $L$ depends on $a^{[3]}$
        - $a^{[3]}$ depends on $z^{[3]}$
        - $z^{[3]}$ is related to $a^{[2]}$
        - $a^{[2]}$ depends on $z^{[2]}$
        - $z^{[2]}$ depends on $W^{[2]}$
        - Therefore, $\frac{dL}{dW^{[2]}} = \frac{dL}{da^{[3]}} \cdot \frac{da^{[3]}}{dz^{[3]}} \cdot \frac{dz^{[3]}}{da^{[2]}} \cdot \frac{da^{[2]}}{dz^{[2]}} \cdot \frac{dz^{[2]}}{dW^{[2]}}$
    - After more math, $\frac{dL}{dW^{[2]}} = (a^{[3]}-y)W^{[3]}g'(z^{[2]})a^{[1]T}$
    - Essentially, $W^{[3]}$ and $a^{[3]}$ propogate backward to make changes to $W^{[2]}$
- Can use full gradient descent, mini-batch gradient descent, or momentum
- L2 regularize with descent step $W=(1-\alpha \lambda)W - \alpha \frac{dJ}{dW}$


# Techniques

## Cross Validation

- Hold out validation:  
&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2591;&#x2591;
    - Split data into 2 groups. Measure performance with test set.
    - For small datasets this may be inefficient use of data.
    - Computationally efficient
- K-fold cross validation:  
&#x2591;&#x2591;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;  
&#x2593;&#x2593;&#x2591;&#x2591;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;  
&#x2593;&#x2593;&#x2593;&#x2593;&#x2591;&#x2591;&#x2593;&#x2593;&#x2593;&#x2593;  
&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2591;&#x2591;&#x2593;&#x2593;  
&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2591;&#x2591;
    - Use $k-1$ folds to train, use remaining folds to test. Iterate through each configuration. Model performance is usually the mean of each configuration performance.
    - More reliable than hold ou but $k$-times more expensive.
- Leave-one-out cross validation (LOOCV):  
&#x2591;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;  
&#x2593;&#x2591;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;  
&#x2593;&#x2593;&#x2591;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;  
$\quad\vdots$  
&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2591; 
    - k-fold where $k=m$, from m samples
    - Commonly used when m is small
- 3-way hold out cross validation:  
&#x2591;&#x2591;&#x2592;&#x2592;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;&#x2593;
    - For model selection
    - Include a validation set.
    - Select hyperparameters/$\theta$ with validation set, then test with test set.
    - Can do nested cross validation, where you use k-fold to select test set and a nested k-fold to select hyperparameters/$\theta$

| | Large Dataset | Small Dataset |
|-|-|-|
| Estimating Performance | 2-way hold out | k-fold, LOOCV |
| Model Selection, Hyperparameter Optimization & Performance Estimation | 3-way hold out | LOOCV with an independent test set |
| Model Comparison | Disjoint training sets and test sets | Nested Cross validation |

---

## Feature Selection

- In the case where n >> m and some features in n might not be relevant
- Wrapper Methods: Wrap your algorithm in another algorithm with subsets of features. Algorithm dependent. More computationally intensive.
    - Forward Search:  
        - Sequential variation:  
        Start with an empty feature set $f=\emptyset$  
        Repeat for $i=1,2,\dots,n$  
            > Let $f_i=f\cup\{i\}$  
            Train algorithm using features $x_j, j\in f_i$  
            Meaure generalization error for $f_i$  
            $f=f_i$  
        - Choose $f$ as the lowest generalization across all $i$ variations
        - Performs best when optimal $f$ is small
        - Sequential forward search is unable to remove specific features that are less useful/obsolete
    - Backward Search:
        - Sequential variation:  
        Start with all features $f=\{1,2,\dots,n\}$  
        Repeat for $i=n,n-1,\dots,1$  
            > Let $f_i=f$  
            Train algorithm using features $x_j, j\in f_i$  
            Meaure generalization error for $f_i$  
            $f=f_i \setminus i$  
        - Performs best when optimal $f$ is large
        - Sequential backward search cannot reinstant useful features once they are out
- Filter Methods: Compute score on each feature that measures how informative it is. Algorithm independent. Less computationally intensive.
    - Define some score $S(j)$ that measures how informative column $j$ is, then pick the $k$ features with the larges score.
    - Scoring functions:
        - Absolute value of the correlation between $x_j$ and $y$
        - Mutual information between $x_j$ and $y$  
        $MI(x_j,y)=\displaystyle\sum_{x_j\in\{0,1\}}\displaystyle\sum_{y\in\{0,1\}}p(x_j,y)\log\frac{p(x_j,y)}{p(x_j)p(y)}$
    - Can choose $k$ using cross-validation
- Other methods:
    - Remove features with low variance
    - Tree-based estimators

---

## Principal Components Analysis (PCA)

- Given a dataset where $x^{(i)}\in\mathbb{R}^n$, find a subspace $\mathbb{R}^k$, k << n, in which $x$ approximately lies
- Remove redundant features, project data into a difference subspace
- Preprocessing: Normalize the dataset
- Find direction of maximum variance  
    - Consider a new unit vector $\bar u$ where $||u||=1$. $proj_{\bar u}(x^{(i)})=x^{(i)T} \bar u$
    - Maximize $\frac{1}{m}\displaystyle\sum_{i=1}^{m}(x^{(i)T} \bar u)^2$ $\rightarrow$ $\max \bar u ^{T}\Sigma u $ such that $\bar u^T \bar u=1$
        - This has n solutions for eignvalues $\lambda _1, \lambda _2, \dots, \lambda _n$
        - The principal eigenvector is $u_1$ where $\Sigma u_1 = \lambda _1 u_1$
    - $u_1$ and $u_2$ are orthogonal
        
---

## Regularization

- Add a weight to unimportant $\theta$ in the cost function to encourage small $\theta$
- For linear or logistic regression
- Select $\lambda$ using cross-validation
- Cost function: $J(\theta)=\dots\frac{1}{m}\left((\text{measure of fit})+\lambda\displaystyle\sum_{j=1}^{n}\theta_{j}^{2}\right)$ except for $\theta_{0}$
- Closed form solution for linear regression: 
```math
\theta=\left(X^{T}X+\lambda
\begin{bmatrix}
0 & & & &\\
& 1 & & &\\
& & 1 & &\\
& & & \ddots &\\
& & & & 1
\end{bmatrix}
\right)^{-1}X^{T}\bar{y}
```
- Gradient descent: $\theta_{j}=\theta_{j}-\alpha(\frac{1}{m}\left((\text{derivative of measure of fit})+\lambda\theta_{j}\right))$ except for $\theta_{0}$