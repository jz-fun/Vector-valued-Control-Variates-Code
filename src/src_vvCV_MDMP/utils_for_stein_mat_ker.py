import torch
import numpy as np
from src.src_vvCV_MD1P.score_funcs import *
from src.src_vvCV_MD1P.stein_operators import *
from src.src_vvCV_MD1P.base_kernels import *


##########################################################################################
# For multiple datasets: SGD on sum of log marginal likelihood
##########################################################################################



class negative_log_marginal_lik_MRI_multidats_mat_kernel(torch.nn.Module):
    def __init__(self, prior_kernel, base_kernel, batch_size,  Xs_tensor, Ys_tensor, scores_Tensor, flag_if_use_medianheuristic):
        """
        :param prior_kernel:   a class; note here even for the most general case vv-CV-MDMP, we still use stein operator for scalar-valued kernel to tune
        :param base_kernel:    a class; a scalar-valued kernel
        :param Xs_tensor:      3d Tensor, T * m * d ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param Ys_tensor:      3d Tensor, T * m * 1 ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param scores_Tensor:  4d Tensor, T * m * T * d ; That is, we have T tasks, each has a dataset with sample size m. For each instance, we need a 2d Tensor of size T*d.
        :param flag_if_use_medianheuristic: whether or not to use median heuristic
        """
        super(negative_log_marginal_lik_MRI_multidats_mat_kernel, self).__init__()

        T = Xs_tensor.size()[0]
        m = Xs_tensor.size()[1]
        d = Xs_tensor.size()[2]

        Xs_flattened = Xs_tensor.reshape(T*m, -1) # 2d tensor; (Tm)*d; all datasets are stacked along rows

        # Kernel hyper-parameters
        if flag_if_use_medianheuristic == True:
            if isinstance(base_kernel(), rbf_kernel) == True:  # For rbf To avoid over-parametrization, set outputscale to 1
                self.base_kernel_parm1_raw = torch.log(torch.ones(1, dtype=torch.float32, requires_grad=False))
            else:
                self.base_kernel_parm1_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True)))

            self.base_kernel_parm2_raw = torch.log(torch.ones(1, dtype=torch.float32, requires_grad=False) *  torch.median(torch.cdist(Xs_flattened, Xs_flattened, p=2) ** 2) / torch.log(Xs_flattened.size()[0] * torch.ones(1))  )# Use median heuristic for the lengthscale


        if flag_if_use_medianheuristic == False:
            if isinstance(base_kernel(), rbf_kernel) == True:   # For rbf To avoid over-parametrization, set outputscale to 1
                self.base_kernel_parm1_raw = torch.log(torch.ones(1, dtype=torch.float32, requires_grad=False))
            else:
                self.base_kernel_parm1_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True)))  #

            self.base_kernel_parm2_raw = torch.nn.Parameter( torch.log( torch.ones(1, dtype=torch.float32, requires_grad=True) * 10 )) #10


        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel

        self.T = T
        self.m = m
        self.d = d

        self.Xs_tensor = Xs_tensor
        self.Ys_tensor = Ys_tensor
        self.scores_Tensor = scores_Tensor

        self.batch_size = batch_size



    def forward(self, batch_sample_indices, beta_cstkernel = 1):

        assert len(batch_sample_indices)  ==  self.batch_size

        # Ensure positivity
        self.base_kernel_parm1 = torch.exp(self.base_kernel_parm1_raw)
        self.base_kernel_parm2 = torch.exp(self.base_kernel_parm2_raw)

        # Instantialized the class
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2


        log_mll = 0.
        for idx in range(self.T):
            Y_l = self.Ys_tensor[idx,]             # 2d tensor, m * 1, the lth dataset
            X_l = self.Xs_tensor[idx,]             # 2d tensor, m * d, the lth dataset
            score_X_l = self.scores_Tensor[idx,]   # 3d tensor, m * T * d, the scores for all samples in the current dataset   # Recall that self.scores_Tensor is a 4d Tensor of size T * m * T * d

            Y_l_b = Y_l[batch_sample_indices,:]    # b * 1
            X_l_b = X_l[batch_sample_indices,:]    # b * d
            score_X_l_b = score_X_l[batch_sample_indices, idx, :]   # b * d. Note that we only need the idx_th row.

            k_Xlb_Xlb = kernel_obj.cal_stein_base_kernel(X_l_b, X_l_b, score_X_l_b, score_X_l_b) + beta_cstkernel



            cond_number_threshold = 1e6
            lam = 1e-6

            bad_cond = 1. / np.linalg.cond(k_Xlb_Xlb.detach().numpy()) < 10**(-6)
            k_Ylb = k_Xlb_Xlb + lam * torch.eye(X_l_b.size()[0])
            while bad_cond:
                lam = 10 * lam
                k_Ylb = k_Xlb_Xlb + lam * torch.eye(X_l_b.size()[0])
                bad_cond = np.linalg.cond(k_Ylb.detach().numpy()) >= cond_number_threshold
            k_Ylb.to(dtype=torch.float64)

            if Y_l_b.dim() == 1:
                Y_l_b = Y_l_b.unsqueeze(dim=1)  # ensure Y is a column vector

            distrib = torch.distributions.MultivariateNormal(torch.zeros(Y_l_b.size()[0]), covariance_matrix=k_Ylb)

            log_mll += distrib.log_prob(Y_l_b.squeeze())

        neg_log_mll = -1. * log_mll
        return neg_log_mll








class TuneKernelParams_mllk_MRI_multidats_mat_kernel(object):

    def __init__(self, prior_kernel, base_kernel, Xs_tensor, Ys_tensor, scores_Tensor):
        """

        :param prior_kernel: a class
        :param base_kernel:  a class
        :param Xs_tensor:      3d Tensor, T * m * d ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param Ys_tensor:      3d Tensor, T * m * 1 ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param scores_Tensor:  4d Tensor, T * m * T * d ; That is, we have T tasks, each has a dataset with sample size m. For each instance, we need a 2d Tensor of size T*d.
        """

        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel

        self.Xs_tensor = Xs_tensor
        self.Ys_tensor = Ys_tensor
        self.scores_Tensor = scores_Tensor



    # split an iterable of items into batches
    def chunks(self, ls, batch_size):
        """
        Yield successive n-sized chunks from l.
        :params[in]: ls, an iterable of items
        :params[in]: batch_size, an integer, batch size
        returns a generator
        """
        for i in range(0, len(ls), batch_size):
            yield ls[i:i + batch_size]



    def do_optimize_logmll(self, batch_size, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.001, epochs=100, verbose=True):
        neg_mll = negative_log_marginal_lik_MRI_multidats_mat_kernel(self.prior_kernel, self.base_kernel, batch_size,  self.Xs_tensor, self.Ys_tensor, self.scores_Tensor, flag_if_use_medianheuristic)
        optimizer = torch.optim.Adam(neg_mll.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)


        m = self.Xs_tensor.size()[1]
        train_indices = list(range(m))

        for i in range(epochs):

            batches_generator = self.chunks(train_indices, batch_size)  # this creates a generator

            for batch_idx, batch_indices in enumerate(batches_generator):

                scheduler.step()
                optimizer.zero_grad()
                out = neg_mll(batch_indices, beta_cstkernel)
                out.backward()
                optimizer.step()



            if verbose:
                print(i + 1, iter, out, neg_mll.base_kernel_parm1.detach(),neg_mll.base_kernel_parm2.detach())

            # Random shuffle
            np.random.shuffle(train_indices)

        self.neg_mll = neg_mll





# A helper function
def helper_get_scores(samples, mean_Gaussians, cov_Gaussians):
    """
    :param samples: 2d Tensor of size [m, d]
    :param mean_Gaussian: a tuple of 2d Tensor of size [d, 1]; mean (vector) of a (multivariate) Gaussian distribution
    :param cov_Gaussian:  a tuple of 2d Tensor of size [d, d]; cov-variance (matrix) of a (multivariate) Gaussian distribution
    :param randn_seeds_tuple: a tuple of random seeds of size T, e.g., (1,2,3,4,... T).
                                This is for reproducing the experiments.
                                (which is used for generating random datasets inside the algorithm)
    :return: all cross-over scores for every sample in 'samples'.
    """

    T = len(mean_Gaussians)
    _ = len(cov_Gaussians)
    assert T == _ , "Contain different number of mean vectors and cov-matrices. Check your input."

    d = mean_Gaussians[0].size(0)
    m = samples.size(0)

    score_tensor_all = torch.zeros(m, T, d)


    for j in range(T):
        score_tensor_all[:, j, :] = multivariate_Normal_score(mean_Gaussians[j], cov_Gaussians[j], samples)

    return  score_tensor_all













