import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  N = X.shape[0]
  C = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  for k in range(N):
    scores        = X[k].dot(W)              # (10,)
    exp           = np.exp(scores)           # (10,)
    norm          = np.sum(exp)              # scalar
    softmax       = exp / norm               # (10,)
    correct_pred  = softmax[y[k]]            # scalar [0, 1]
    loss         += -np.log(correct_pred)    # scalar
   
    # Gradient:
    y_oh = np.zeros(C)
    y_oh[y[k]] = 1.0
    dnum = X[k].reshape(-1,1).dot(y_oh*exp.reshape(1,-1))
    dden = X[k].reshape(-1,1).dot((-1/norm**2)*exp.reshape(1,-1))
    dW += -1/correct_pred * (dden*exp[y[k]] + dnum/norm)
       
  loss /= N
  loss += reg * np.sum(W * W)  
    
  dW /= N
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  N = X.shape[0]
  C = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores  = X.dot(W)
  exp     = np.exp(scores)
  norm    = np.sum(exp, axis=1)  
  softmax = exp / norm.reshape(-1,1)
    
  loss    = - scores[range(N),y] + np.log(norm)
  loss    = np.sum(loss, axis=0) / N
  loss   += reg * np.sum(W * W)

  # Gradient:
  dscores = np.zeros_like(scores)
  dscores[range(N), y] = 1.0
          
  dW    -= X.T.dot(dscores) / N
  dW    += X.T.dot(softmax) / N
  dW    += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

