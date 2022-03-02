from src.src_vvCV_MD1P.base_kernels import *





class stein_matrix_valued_kernel(object):
    """
     This is defined for K(x,y) = B k(x,y)
    """
    def __init__(self, base_kernel, T):
        self._base_kernel_parm1 = 1
        self._base_kernel_parm2 = 1
        self.base_kernel = base_kernel      # This is a scalar-valued kernel
        self.T = T

        self._B = torch.diag(torch.ones(T))

    @property
    def base_kernel_parm1(self):
        return self._base_kernel_parm1
    @base_kernel_parm1.setter
    def base_kernel_parm1(self, x):
        self._base_kernel_parm1 = x


    @property
    def base_kernel_parm2(self):
        return self._base_kernel_parm2
    @base_kernel_parm2.setter
    def base_kernel_parm2(self, x):
        self._base_kernel_parm2 = x


    @property
    def B(self):
        return self._B
    @B.setter
    def B(self, x):
        assert torch.det(x) >= 0, "B should be (semi-)PSD. Check the optimisation algorithm."
        self._B = x


    def cal_pairwise(self, x, y, score_mat_x, score_mat_y):
        """
        :param x: 1d Tensor, size d
        :param y: 1d Tensor, size d
        :param T: a value, the matrix-valued kernel function return a T-by-T matrix for each pair of points
        :param score_mat_x: 2d Tensor, size T * d
        :param score_mat_y: 2d Tensor, size T * d
        :return: T-by-T valued K_0(x, y) for a pair of points, and the original K(x,y) = B k(x,y)
        """
        assert self.T == score_mat_x.size(0), "T and score_max_x should match. Check the dimension of score_mat_x."
        assert self.T == score_mat_y.size(0), "T and score_max_y should match. Check the dimension of score_mat_y."

        # Initialized the base_kernel
        base_kernel_obj = self.base_kernel()
        base_kernel_obj.kernel_parm1 = self.base_kernel_parm1
        base_kernel_obj.kernel_parm2 = self.base_kernel_parm2

        # Get the derivatives (given the current values of kernel hyperparameters)
        ker_eval_xy, grad_ker_x, grad_ker_y, gradgrad_k = base_kernel_obj.deriv_base_kernel(x, y)

        # Calcualte
        out = torch.zeros(self.T, self.T)
        for i in range(self.T):
            for j in range(self.T):
                out[i,j] = self.B[i,j] * (gradgrad_k + score_mat_y[j,:] @ grad_ker_x + score_mat_x[i,:] @ grad_ker_y +  ker_eval_xy * (score_mat_y[j,:] @ score_mat_x[i,:] ) )

        return out



    def cal_datasetwise(self, X, Z, score_tensor_X, score_tensor_Z):
        """
        :param X: 2d Tensor; size m * d
        :param Z: 2d Tensor; size n * d
        :param score_tensor_X: 3d Tensor; m * T * d
        :param score_tensor_Z: 3d Tensor; n * T * d
        :return: 4d Tensor; size m * n * T * T
        """
        assert X.size(1) == Z.size(1), "Dimention of samples in two dataset donot match."
        assert score_tensor_X.size(1) == self.T, "Score matrix for each sample should be a T-by-d matrix."
        assert score_tensor_Z.size(1) == self.T, "Score matrix for each sample should be a T-by-d matrix."

        m = X.size(0)
        n = Z.size(0)
        d = X.size(1)

        out = torch.zeros(m, n, self.T, self.T)

        for i in range(m):
            for j in range(n):
                out[i,j] = self.cal_pairwise(X[i], Z[j], score_tensor_X[i,:,:], score_tensor_Z[j, :, :])

        return out



