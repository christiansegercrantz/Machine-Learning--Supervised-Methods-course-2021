## ####################################################
import numpy as np
from sklearn.datasets import load_breast_cancer

# load the data
X, y = load_breast_cancer(return_X_y=True)  ## X input, y output
print(X.shape, y.shape)
## to convert the {0,1} output into {-1,+1}
y = 2 * y -1

mdata,ndim=X.shape                                   ## size of the data 

iscale = 1   ## =0 no scaling, =1 scaling the by the maximum absolute value
if iscale == 1:
  X /= np.outer(np.ones(mdata),np.max(np.abs(X),0))

niter = 10 ## number of iteration 

## initialize eta, lambda for the primal algorithm
eta=0.1              ##  step size
xlambda=0.01          ## balancing constant between loss and regularization
## set the penalty constant for the dual algorithm
C = 1000


