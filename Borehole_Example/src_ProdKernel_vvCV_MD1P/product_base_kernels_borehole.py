import torch
import numpy as np
import math
from matplotlib import pyplot as plt
import time



class prod_rbf_kernel_Borehore(object):
   # in example, lengthscale is a scalar.
    def __init__(self):
        self._kernel_parm1 = torch.ones(1) # convert to torch
        self._kernel_parm2 = torch.ones(1)
        self._kernel_parm3 = torch.ones(1)
        self._kernel_parm4 = torch.ones(1)
        self._kernel_parm5 = torch.ones(1)
        self._kernel_parm6 = torch.ones(1)
        self._kernel_parm7 = torch.ones(1)
        self._kernel_parm8 = torch.ones(1)

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

    @property
    def kernel_parm3(self):
       return self._kernel_parm3
    @kernel_parm3.setter
    def kernel_parm3(self, x):
        self._kernel_parm3 = x


    @property
    def kernel_parm4(self):
       return self._kernel_parm4
    @kernel_parm4.setter
    def kernel_parm4(self, x):
        self._kernel_parm4 = x


    @property
    def kernel_parm5(self):
        return self._kernel_parm5
    @kernel_parm5.setter
    def kernel_parm5(self, x):
        self._kernel_parm5 = x


    @property
    def kernel_parm6(self):
        return self._kernel_parm6
    @kernel_parm6.setter
    def kernel_parm6(self, x):
        self._kernel_parm6 = x


    @property
    def kernel_parm7(self):
        return self._kernel_parm7
    @kernel_parm7.setter
    def kernel_parm7(self, x):
        self._kernel_parm7 = x


    @property
    def kernel_parm8(self):
        return self._kernel_parm8
    @kernel_parm8.setter
    def kernel_parm8(self, x):
        self._kernel_parm8 = x



    def deriv_base_kernel(self, x, y):
        # return relevant derivatives of a pair of points including base kernel value at this pair
        # x: 1d tensor
        # y: 1d tensor
        ## difference between two tensors
        self.dim = x.size()[0] # as x is 1-d tensor here, so size[0] is its dim
        x_minus_y = x - y

        ## Compute k_eval_vec = [k_1(x1, y1) ... kd(xd, yd)]
        diag_vec = torch.cat((self.kernel_parm1, self.kernel_parm2, self.kernel_parm3, self.kernel_parm4, self.kernel_parm5, self.kernel_parm6, self.kernel_parm7, self.kernel_parm8))
        k_eval_vec = torch.exp( -0.5 *x_minus_y.pow(2)/(diag_vec) )
        assert k_eval_vec.size(0) == 8

        ## partial derivative of kernel w.r.t. x
        ker_x =  k_eval_vec.prod() * ((-1. * x_minus_y)/diag_vec)
        assert ker_x.size(0) == 8

        ## partial derivative of kernel w.r.t. y
        ker_y =  k_eval_vec.prod() * (x_minus_y/diag_vec)
        assert ker_y.size(0) == 8

        ## second derivative w.r.t. x and y // laplace
        ker_xy_vec = (1. / diag_vec - ((x_minus_y)**2) / (diag_vec**2)) * k_eval_vec.prod()
        assert ker_xy_vec.size(0) == 8
        ker_xy =  torch.sum(ker_xy_vec)

        ## return these items in the base kernel
        return (k_eval_vec.prod(), ker_x, ker_y, ker_xy)



    def cal_kernel_pairwise(self, x1, x2):
        assert x1.size() == x2.size()
        assert x1.size()[1] == torch.tensor(1)  # ensure a column vector
        diag_vec = torch.cat((self.kernel_parm1, self.kernel_parm2, self.kernel_parm3, self.kernel_parm4, self.kernel_parm5, self.kernel_parm6, self.kernel_parm7, self.kernel_parm8))
        S_inv = torch.diag(1/diag_vec)
        o = torch.exp(-0.5 * (x1-x2).t() @ S_inv @ (x1-x2))
        return o




    def cal_kernel(self, X1, X2):
        m = X1.size()[0]
        n = X2.size()[0]
        prior_covariance = torch.zeros(m,n)
        for i in range(m):
            for j in range(n):
                prior_covariance[i,j] = self.cal_kernel_pairwise(X1[i,].t().unsqueeze(dim=1), X2[j,].t().unsqueeze(dim=1))  # ZS: such operations is to ensure a column vector for each
        return prior_covariance





