import torch
import numpy as np
import math
from matplotlib import pyplot as plt


def product_Normal_score(mus, vars, X):
    """
    calculate score value for all points in X,
           assuming each point x_i, whose element x_ij comes from N(mu_j, var_j)
           i.e. x_i \sim \prod_j=1^d N(mu_j, var_j)

    :param mus: d * 1, each element is a mean for each Gaussian inside the product of Gaussian
    :param vars: d * 1, each element is a variance for each Gaussian inside the product of Gaussian
    :param X: 2d tensor, n * d, each row is a point
    :return: 2d tensor, n * d, each row is the grad_x logp(x)
    """
    assert mus.dim()  == 2 #
    assert vars.dim() == 2
    assert X.size()[1] == mus.size()[0] # should equal, the dim of a instance

    n = X.size()[0]
    d = X.size()[1]
    out = torch.zeros(n,d)
    for i in range(n):
        x_i = X[i].squeeze().unsqueeze(dim=1)
        assert x_i.size() == mus.size()

        for j in range(d):
            out[i, j] = (-1. * (x_i[j] - mus[j])) / vars[j]

    return out

