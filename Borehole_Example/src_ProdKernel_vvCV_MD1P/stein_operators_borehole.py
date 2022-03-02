from Borehole_Example.src_ProdKernel_vvCV_MD1P.score_funcs_borehole import *





class stein_base_kernel_borehole(object):

    def __init__(self, base_kernel): #, beta=1
        """
        :param distribution: an object from some distribution class
        :param base_kernel: a base kernel class,  here is prod_rbf_kernel_Borehore class
        :param base_kernel_parm1:
        :param base_kernel_parm2:
        """

        self._base_kernel_parm1 = torch.ones(1)
        self._base_kernel_parm2 = torch.ones(1)
        self._base_kernel_parm3 = torch.ones(1)
        self._base_kernel_parm4 = torch.ones(1)
        self._base_kernel_parm5 = torch.ones(1)
        self._base_kernel_parm6 = torch.ones(1)
        self._base_kernel_parm7 = torch.ones(1)
        self._base_kernel_parm8 = torch.ones(1)
        self.base_kernel = base_kernel


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
    def base_kernel_parm3(self):
        return self._base_kernel_parm3
    @base_kernel_parm3.setter
    def base_kernel_parm3(self, x):
        self._base_kernel_parm3 = x


    @property
    def base_kernel_parm4(self):
        return self._base_kernel_parm4
    @base_kernel_parm4.setter
    def base_kernel_parm4(self, x):
        self._base_kernel_parm4 = x



    @property
    def base_kernel_parm5(self):
        return self._base_kernel_parm5
    @base_kernel_parm5.setter
    def base_kernel_parm5(self, x):
        self._base_kernel_parm5 = x


    @property
    def base_kernel_parm6(self):
        return self._base_kernel_parm6
    @base_kernel_parm6.setter
    def base_kernel_parm6(self, x):
        self._base_kernel_parm6 = x


    @property
    def base_kernel_parm7(self):
        return self._base_kernel_parm7
    @base_kernel_parm7.setter
    def base_kernel_parm7(self, x):
        self._base_kernel_parm7 = x


    @property
    def base_kernel_parm8(self):
        return self._base_kernel_parm8
    @base_kernel_parm8.setter
    def base_kernel_parm8(self, x):
        self._base_kernel_parm8 = x



    def cal_stein_base_kernel(self, X, Z, score_tensor_X, score_tensor_Z):
        """
        :param X: 2d tensor, m * p matrix
        :param Z: 2d tensor, n * p matrix
        :param score_tensor_X: 2d tensor, m * d
        :param score_tensor_Z: 2d tensor, n * d
        :return: kernel matrix, k_0(X, Z), m * n
        """
        # instantialized the kernel class
        base_kernel_obj = self.base_kernel()
        base_kernel_obj.kernel_parm1 = self.base_kernel_parm1
        base_kernel_obj.kernel_parm2 = self.base_kernel_parm2
        base_kernel_obj.kernel_parm3 = self.base_kernel_parm3
        base_kernel_obj.kernel_parm4 = self.base_kernel_parm4
        base_kernel_obj.kernel_parm5 = self.base_kernel_parm5
        base_kernel_obj.kernel_parm6 = self.base_kernel_parm6
        base_kernel_obj.kernel_parm7 = self.base_kernel_parm7
        base_kernel_obj.kernel_parm8 = self.base_kernel_parm8


        grad_logpX = score_tensor_X
        grad_logpZ = score_tensor_Z

        # einsum -- https://rockt.github.io/2018/04/30/einsum
        grad_k_X = torch.zeros(X.size()[0], Z.size()[0], X.size()[1])
        grad_k_Z = torch.zeros(X.size()[0], Z.size()[0], X.size()[1])
        gradgrad_k = torch.zeros(X.size()[0], Z.size()[0])
        for i in range(X.size()[0]):
            for j in range(Z.size()[0]):
                _, grad_k_X[i, j, :], grad_k_Z[i, j, :], gradgrad_k[i, j] = base_kernel_obj.deriv_base_kernel(X[i], Z[j])

        a = gradgrad_k
        b = torch.einsum('ik,ijk -> ij' , grad_logpX, grad_k_Z)
        c = torch.einsum('jk,ijk -> ij' , grad_logpZ, grad_k_X)
        d = (grad_logpX @ grad_logpZ.t()) * base_kernel_obj.cal_kernel(X,Z)


        self.grad_k_X = grad_k_X
        self.grad_k_Z = grad_logpZ
        self.grad_logpX = grad_logpX
        self.grad_logpZ = grad_logpZ

        value_stein_rbf_kernel = a + b + c + d    # value_stein_rbf_kernel = self.beta + a + b + c + d

        return value_stein_rbf_kernel



# This is a helper function to verify the above 'stein_base_kernel_borehole'
class deriv_prod_rbf_kernel():
    # calculate pair-wise derivative of deri_x1 k(x1, x2) and deriv_x2 k(x1, x2)
    def __init__(self, l_sq_tensor):
        """
        param l_sq_tensor: 2d tensor: d * 1,  each element of which is the squred lengthscale
        """
        self.dim_x = l_sq_tensor.size()[0]
        self.l_sq_tensor = l_sq_tensor

    def deriv_x1_k(self, x1, x2):
        """
        :param x1: 2d tensor, d * 1
        :param x2: 2d tensor, d * 1
        :return:
        """
        vec_ker_comps = torch.zeros(self.dim_x, 1)
        for i in range(self.dim_x):
            vec_ker_comps[i] = torch.exp((x1[i] - x2[i]) ** 2 / (-2 * self.l_sq_tensor[i]))
        print(vec_ker_comps)

        o = torch.zeros(self.dim_x, 1)
        for j in range(self.dim_x):
            vec_ker_comps_oim_curr = torch.cat([vec_ker_comps[:j], vec_ker_comps[j + 1:]])
            o[j] = torch.exp((x1[j] - x2[j]).pow(2) / (-2 * self.l_sq_tensor[j])) * ((2 * (x1[j] - x2[j])) / (-2 * self.l_sq_tensor[j])) * vec_ker_comps_oim_curr.prod()

        return o

    def deriv_x2_k(self, x1, x2):
        o = -1. * self.deriv_x1_k(x1, x2)
        return o









