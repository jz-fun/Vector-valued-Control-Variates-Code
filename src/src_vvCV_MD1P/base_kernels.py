

import torch
import numpy as np
import math
from matplotlib import pyplot as plt
import time



##### 1. Standard kernels for \Pi is defined on R^d #########
class base_kernel_2(object):
    # : the one used in Oates et al JRSSB 2017

    def __init__(self):
        self._kernel_parm1 = 1
        self._kernel_parm2 = 1

    @property
    def kernel_parm1(self):
        return self._kernel_parm1
    @kernel_parm1.setter
    def kernel_parm1(self, x):
        self._kernel_parm1 = x


    @property
    def kernel_parm2(self):
        return self._kernel_parm2
    @kernel_parm2.setter
    def kernel_parm2(self, x):
        self._kernel_parm2 = x


    def deriv_base_kernel(self, x, y):
        # return relevant derivatives of a pair of points including base kernel value at this pair
        # x: 1d tensor
        # y: 1d tensor
        ## difference between two tensors
        self.dim = x.size()[0] # ZS: as x is 1-d tensor here, so size[0] is its dim
        x_minus_y = x - y
        ## quadratic terms for x
        quad_x = (1. + self.kernel_parm1 * x.pow(2).sum()).pow(-1)
        ## quadratic terms for y
        quad_y = (1. + self.kernel_parm1 * y.pow(2).sum()).pow(-1)
        ## evaluation of kernel function at this pair of points k(x,y)
        ker_eval = quad_x * quad_y * torch.exp(-x_minus_y.pow(2).sum() / 2 \
                                                   / (self.kernel_parm2 ** 2))
        ## partial derivative of kernel w.r.t. x
        ker_x = ker_eval * (-2 * self.kernel_parm1 * x * quad_x - x_minus_y / self.kernel_parm2 ** 2)
        ## partial derivative of kernel w.r.t. y
        ker_y = ker_eval * (-2 * self.kernel_parm1 * y * quad_y + x_minus_y / self.kernel_parm2 ** 2)
        ## second derivative w.r.t. x and y // laplace
        ker_xy = ker_eval * (4 * (self.kernel_parm1 ** 2) * quad_x * quad_y * torch.matmul(x, y) + \
                                 2 * self.kernel_parm1 / (self.kernel_parm2 ** 2) * quad_y * torch.matmul((x - y), y) - \
                                 2 * self.kernel_parm1 / (self.kernel_parm2 ** 2) * quad_x * torch.matmul((x - y), x) - \
                                 1 / (self.kernel_parm2 ** 4) * torch.matmul((x - y), (x - y)) + self.dim / (
                                             self.kernel_parm2 ** 2))
        ## return these items in the base kernel
        return (ker_eval, ker_x, ker_y, ker_xy)


    def cal_kernel(self, X1, X2):
        if len(X1.size()) == 1:   #: as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
            X1 = X1.unsqueeze(1)
        if len(X2.size()) == 1:
            X2 = X2.unsqueeze(1)

        dist_mat = torch.cdist(X1, X2, p=2)**2

        m = X1.size()[0]
        n = X2.size()[0]
        mat = torch.zeros(m,n)

        norms_X1 = X1.norm(dim = 1, p=2).pow(2)  # as we assume each row represents a point, we compute norm by rows.
        norms_X2 = X2.norm(dim = 1, p=2).pow(2)

        norms_X1 = norms_X1.unsqueeze(dim=1) # size is [m,1]
        norms_X2 = norms_X2.unsqueeze(dim=0) # size is [1,n]

        mat = (1 + self.kernel_parm1 * norms_X1.repeat(1,n)) * (1 + self.kernel_parm1 *  norms_X2.repeat(m,1))

        prior_covariance = (1/(mat)) * torch.exp(-0.5 * dist_mat/self.kernel_parm2**2)
        return prior_covariance










class polynomial_kernel(object):
    def __init__(self):
        self._kernel_parm1 = 1  # By default, this is the degree of the polynomial
        self._kernel_parm2 = 1  # By default, this is the bias constant


    @property
    def kernel_parm1(self):
        return self._kernel_parm1
    @kernel_parm1.setter
    def kernel_parm1(self, x):
        self._kernel_parm1 = x


    @property
    def kernel_parm2(self):
        return self._kernel_parm2
    @kernel_parm2.setter
    def kernel_parm2(self, x):
        self._kernel_parm2 = x



    def deriv_base_kernel(self, x, y):
        # return relevant derivatives of a pair of points including base kernel value at this pair
        # x: 1d tensor
        # y: 1d tensor
        ## difference between two tensors
        self.dim = x.size()[0] # as x is 1-d tensor here, so size[0] is its dim

        x_dot_y = x @ y

        ## evaluation of kernel function at this pair of points k(x,y)
        ker_eval = (x_dot_y + self.kernel_parm2)**self.kernel_parm1


        ## partial derivative of kernel w.r.t. x
        ker_x = self.kernel_parm1 * ((x_dot_y + self.kernel_parm2)**(self.kernel_parm1 - 1)) * y
        assert ker_x.size(0) == self.dim

        ## partial derivative of kernel w.r.t. y
        ker_y =  self.kernel_parm1 * ((x_dot_y + self.kernel_parm2)**(self.kernel_parm1 - 1)) * x
        assert ker_y.size(0) == self.dim


        ## second derivative w.r.t. x and y // laplace
        ker_xy = self.kernel_parm1 * (self.kernel_parm1 - 1) * ((x_dot_y + self.kernel_parm2)**(self.kernel_parm1 - 2)) * x_dot_y + self.dim * self.kernel_parm1 * ((x_dot_y + self.kernel_parm2)**(self.kernel_parm1 - 1))


        ## return these items in the base kernel
        return (ker_eval, ker_x, ker_y, ker_xy)


    def cal_kernel(self, X1, X2):
        """
        :param X1:  2d Tensor, m * d
        :param X2:  2d Tensor, n * d
        :return:    2d Tensor, m * n --- Gram matrix
        """
        if len(X1.size()) == 1:   # as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
            X1 = X1.unsqueeze(1)
        if len(X2.size()) == 1:
            X2 = X2.unsqueeze(1)

        assert X1.size(1) == X2.size(1), "Dimension should match."

        dotprod_mat = X1 @ X2.t()
        assert dotprod_mat.size() == torch.Size([X1.size(0), X2.size(0)])

        prior_covariance = (dotprod_mat + self.kernel_parm2) ** self.kernel_parm1

        return prior_covariance









class rbf_kernel(object):
   # In the example, lengthscale l is a scalar.
   #       --- The kernel_parm2 is l^2 (to match the def. of median heuristic of l^2), the postivity of it are assured by exp transformation in the process of tuning them
    def __init__(self):
        self._kernel_parm1 = torch.ones(1) # convert to torch
        self._kernel_parm2 = torch.ones(1)

    @property
    def kernel_parm1(self):
        return self._kernel_parm1
    @kernel_parm1.setter
    def kernel_parm1(self, x):
        self._kernel_parm1 = x


    @property
    def kernel_parm2(self):
       return self._kernel_parm2
    @kernel_parm2.setter
    def kernel_parm2(self, x):
        self._kernel_parm2 = x


    def deriv_base_kernel(self, x, y):
        # return relevant derivatives of a pair of points including base kernel value at this pair
        # x: 1d tensor
        # y: 1d tensor
        ## difference between two tensors
        self.dim = x.size()[0] # ZS: as x is 1-d tensor here, so size[0] is its dim

        x_mins_y = x - y

        ## evaluation of kernel function at this pair of points k(x,y)
        ker_eval = self.kernel_parm1 * torch.exp(x_mins_y @ x_mins_y/(-2*self.kernel_parm2))


        ## partial derivative of kernel w.r.t. x
        ker_x = -1 * ker_eval * x_mins_y/self.kernel_parm2
        assert ker_x.size(0) == self.dim

        ## partial derivative of kernel w.r.t. y
        ker_y =  ker_eval * x_mins_y/self.kernel_parm2
        assert ker_y.size(0) == self.dim


        ## second derivative w.r.t. x and y // laplace
        ker_xy_vec = (1./self.kernel_parm2 - x_mins_y.pow(2)/self.kernel_parm2.pow(2)) * ker_eval
        assert ker_xy_vec.size(0) == self.dim
        ker_xy = ker_xy_vec.sum()

        ## return these items in the base kernel
        return (ker_eval, ker_x, ker_y, ker_xy)


    def cal_kernel(self, X1, X2):
        if len(X1.size()) == 1:   # as we always assume that x's are stacked in rows. but for 1D vectors, better to unsqueeze before enterring the model
            X1 = X1.unsqueeze(1)  # suppose have m points
        if len(X2.size()) == 1:
            X2 = X2.unsqueeze(1)  # suppose have n points
        dist_mat = torch.cdist(X1, X2, p=2)**2    # m by n
        prior_covariance = self.kernel_parm1 * torch.exp(-0.5 * dist_mat / self.kernel_parm2)
        return prior_covariance


