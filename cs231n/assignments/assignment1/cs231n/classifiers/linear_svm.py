from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # dW 表示loss函数的偏导

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  #计算第i个样本示例的在10种分类（10种权重矩阵）下的得分
        correct_class_score = scores[y[i]]  #样本示例i对应正确标签的得分
        for j in range(num_classes):
            if j == y[i]: #此时对应分类标签正确，跳过
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:  #若错误得分比（正确得分-1）要大，增加到loss中
                loss += margin
                dW[:, y[i]] -= X[i].T
                dW[:, j] += X[i].T


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train #loss= data loss
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W) #正则化：loss+=regularization loss
    # 为什么正则化：引入惩罚项，惩罚大的权重项，提高模型的泛化能力。
    # 因为不希望W矩阵中的某一个很大的权重值对预测产生很大的影响。
    # 分类器更希望通过衡量所有输入维度去综合得出分类结果，而不是仅仅考虑某一两个有很高的权重的维度。
    # 我们做的就是找到权重矩阵得到尽可能小的损失值。


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)  #计算样本示例的在10种分类（10种权重矩阵）下的得分
    correct_class_scores = scores[np.arange(num_train), y]  #样本示例对应正确标签的得分
    correct_class_scores = np.reshape(correct_class_scores, (num_train, 1))
    margins = scores - correct_class_scores + 1.0
    margins[np.arange(num_train), y] = 0.0  # 第i个样本示例在对应分类标签处的边界值置为0
    margins[margins<=0] = 0.0 # 将边界值<=0的置为0
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg *np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    margins[margins>0] = 1.0
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -row_sum
    dW += np.dot(X.T, margins)/num_train + reg* W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
