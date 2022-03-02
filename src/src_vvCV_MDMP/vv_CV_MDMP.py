import torch
import numpy as np
from src.src_vvCV_MDMP.utils_for_stein_mat_ker import *
from src.src_vvCV_MDMP.stein_operators_mat_ker import *



class penalized_ls_objective_vectorvaluedfunc_MDMP(torch.nn.Module):
    def __init__(self, optim_base_kernel_parms, prior_kernel, base_kernel, batch_size, Xs_tensor, Ys_tensor, scores_Tensor):
        """
        :param optim_base_kernel_parms:
        :param prior_kernel:  a class; here should be the classes in 'stein_operator_mat_ker', e.g., stein_matrix_valued_kernel
        :param base_kernel:   a class; since we use K = B k, so here this class is scalar-valued kernel class, e.g, base_kernel_2
        :param batch_size:    a number, e.g. 5
        :param Xs_tensor:     3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param Ys_tensor:     3d tensor, T * m * 1 when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param scores_Tensor: 4d Tensor, T * m * T * d ; That is, we have T tasks, each has a dataset with sample size m. For each instance, we need a 2d Tensor of size T*d.
        """
        super(penalized_ls_objective_vectorvaluedfunc_MDMP, self).__init__()

        self.T = Xs_tensor.size()[0]
        self.m = Xs_tensor.size()[1]
        self.d = Xs_tensor.size()[2]
        self.total_no_points = self.T * self.m

        Y_means = Ys_tensor.mean(dim=1)  # 2d tensor; T * 1

        # Initializing raw params for constructing B
        self.paramsB = torch.nn.Parameter(torch.zeros(self.T, self.T, dtype=torch.float32, requires_grad=True))  #

        # Initializing model parameters (theta, c)
        self.theta = torch.nn.Parameter(torch.zeros(self.total_no_points, self.T, dtype=torch.float, requires_grad=True))
        self.c = torch.nn.Parameter(torch.ones(self.T, 1, dtype=torch.float32, requires_grad=True) * Y_means)  # column means are MC estimate for each integral, a p * 1 tensor

        #
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.batch_size = batch_size
        self.Xs_tensor = Xs_tensor
        self.Ys_tensor = Ys_tensor
        self.scores_Tensor = scores_Tensor


        # Set to be the optimized kernel hyper-parameters
        self.base_kernel_parm1 = optim_base_kernel_parms[0].detach()
        self.base_kernel_parm2 = optim_base_kernel_parms[1].detach()


        # Instantialized the class
        kernel_obj = self.prior_kernel(self.base_kernel, self.T)
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2

        self.kernel_obj = kernel_obj

        # Note that parmsB should be multipled to the kernel tensors after being tranformed, which should be included in the 'def forward'.
        # Here we set B to be all ones is to keep all computed elements.
        kernel_obj.B = torch.ones(self.T, self.T)

        # The giant 'Gram_Tensor' is a 6d Tensor, of size [T, T, m, m, T, T]
        Gram_Tensor = torch.zeros(self.T, self.T, self.m, self.m, self.T, self.T)
        for i in range(self.T):
            for j in range(self.T):
                Gram_Tensor[i, j, :, :, :, :] = kernel_obj.cal_datasetwise( self.Xs_tensor[i], self.Xs_tensor[j], self.scores_Tensor[i], self.scores_Tensor[j]   )  #  X, Z, score_tensor_X, score_tensor_Z
        self.Gram_Tensor = Gram_Tensor



    def forward(self, batch_sample_indices, regularizer_const, regularizer_const_FB, ifopt_hyper_parms, ifopt_model_parms):

        assert len(batch_sample_indices) == self.batch_size  #

        if ifopt_hyper_parms:
            self.paramsB.requires_grad_(True)
            self.theta.requires_grad_(False)
            self.c.requires_grad_(False)

        if ifopt_model_parms:
            self.paramsB.requires_grad_(False)
            self.theta.requires_grad_(True)
            self.c.requires_grad_(True)


        # THIS ensures every iteration, self.B is positive semi-definite
        #     --  self.LowerTriAdj_paramsB = L (diagonal elements are forced > 0)
        #     --  self.B = L L^T
        self.LowerTri_paramsB = torch.tril(self.paramsB)
        self.diag_paramsB = torch.exp(torch.diag(self.LowerTri_paramsB))  # ensure diagonal elements > 0 (recall LL^T decomposition of psd matrix)
        mask = torch.diag(torch.ones_like(self.diag_paramsB))  # therefore, if the diagonal elements of self.paramsB are zero intially, then self.B will have diagonal elements equal to 1.
        self.LowerTriAdj_paramsB = mask * torch.diag(self.diag_paramsB) + (1. - mask) * self.LowerTri_paramsB
        self.B = self.LowerTriAdj_paramsB @ self.LowerTriAdj_paramsB.t()

        # AN ESSENTIAL STEP
        self.Gram_Tensor_timedB = self.Gram_Tensor * self.B.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)   #To broadcast, self.B needs to match the dim of self.Gram_tensor, which is [T, T, m, m, T, T]


        # Get batch samples
        Xs_batch = self.Xs_tensor[:, batch_sample_indices, :].reshape(self.T * self.batch_size, -1)  # 2d tensor , (Tb) * d
        Ys_batch = self.Ys_tensor[:, batch_sample_indices, :].reshape(self.T * self.batch_size, -1)  # 2d tensor , (Tb) * 1


        preds = torch.zeros(self.T, self.batch_size)
        for idx in range(self.T):
            Grams_relevant_to_cur_dat = self.Gram_Tensor_timedB[idx]   # 5d Tensor [T, m, m, T, T]
            Component_idx = Grams_relevant_to_cur_dat[:, :, :, idx, :] # 4d Tensor [T, m, m, T]
            middle1 = Component_idx.reshape(self.T, self.m, -1)
            middle2 = middle1.transpose(0,1)
            Component_idx_reshaped = middle2.reshape(self.m, -1)  # This is K_0(X_idx, {X_t}_{t=1}^{T})_{1, \cdot}; 2d Tensor of size m_idx * |m|T

            cur_indices = torch.tensor(batch_sample_indices)
            preds[idx,] = Component_idx_reshaped[cur_indices,] @ self.theta.view(self.total_no_points * self.T) + self.c[idx,]  # This is 1d Tensor of size [b]

        preds = preds.reshape(self.T * self.batch_size, -1)
        assert preds.size() == Ys_batch.size()
        obj_batch = (1. / self.batch_size) * (Ys_batch - preds).pow(2).sum()

        penlty_on_B = torch.norm(self.B, p='fro').pow(2)
        scalable_objective = obj_batch + regularizer_const_FB * penlty_on_B + regularizer_const * torch.norm(self.theta, p='fro').pow(2)

        return scalable_objective.squeeze()








class VV_CV_vectorvaluedfuncs_model_MDMP(object):
    def __init__(self, vv_cv_objective, prior_kernel, base_kernel, Xs_tensor, Ys_tensor, scores_Tensor):
        """
        :param vv_cv_objective:  an objective class. Here should be 'penalized_ls_objective_vectorvaluedfunc_MDMP'
        :param prior_kernel:   a class. Here should be 'stein_matrix_valued_kernel'
        :param base_kernel:    a class; since we use K = B k, so here this class is scalar-valued kernel class, e.g, base_kernel_2
        :param Xs_tensor:     3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param Ys_tensor:     3d tensor, T * m * 1 when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param scores_Tensor: 4d Tensor, T * m * T * d ; That is, we have T tasks, each has a dataset with sample size m. For each instance, we need a 2d Tensor of size T*d.
        """

        self.vv_cv_objective = vv_cv_objective
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.Xs_tensor = Xs_tensor
        self.Ys_tensor = Ys_tensor
        self.scores_Tensor = scores_Tensor



    # Tune kernel hyper-parameters w.r.t. sum of log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        # Note here for tuning, we still use scalar Stein kernel
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_multidats_mat_kernel(stein_base_kernel_MV_2, self.base_kernel, self.Xs_tensor, self.Ys_tensor, self.scores_Tensor)
        tune_kernelparams_negmllk_obj.do_optimize_logmll(batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr, epochs, verbose)
        optim_base_kernel_parms = torch.Tensor([tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm1, tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm2])
        self.optim_base_kernel_parms =optim_base_kernel_parms.detach()
        return optim_base_kernel_parms.detach()


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


    # Optimize B and (Theta, c) via alternating minimization
    def do_optimize_vv_CV(self, regularizer_const = 0.01, regularizer_const_FB = 1, batch_size = 10, lr=0.2, epochs=200, verbose=True):

        fitting_obj = self.vv_cv_objective(self.optim_base_kernel_parms, self.prior_kernel, self.base_kernel, batch_size,  self.Xs_tensor, self.Ys_tensor, self.scores_Tensor)

        optimizer = torch.optim.Adam(fitting_obj.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        saved_BQ_est = torch.zeros(epochs, fitting_obj.T,  1)
        saved_loss  = torch.zeros(epochs, 1)
        m = self.Xs_tensor.size()[1]
        train_indices = list(range(m))

        for i in range(epochs):
            batches_generator = self.chunks(train_indices, batch_size)
            for batch_idx, batch_indices in enumerate(batches_generator):
                # Step 1. minimizing the objective w.r.t. B when fixing theta and c.
                for k in range(1):
                    optimizer.zero_grad()
                    out = fitting_obj(batch_sample_indices = batch_indices, regularizer_const = regularizer_const, regularizer_const_FB = regularizer_const_FB, ifopt_hyper_parms = True, ifopt_model_parms = False)
                    out.backward()
                    optimizer.step()

                    if verbose:
                        print("Current Lr is {}.".format(optimizer.param_groups[0]['lr']))
                        print("AMStage-Opt hyper-parms", i + 1, batch_idx, fitting_obj.c.requires_grad,fitting_obj.theta.requires_grad, fitting_obj.base_kernel_parm1.requires_grad, fitting_obj.base_kernel_parm2.requires_grad, fitting_obj.B.requires_grad)
                        print("Current B is {}.".format(fitting_obj.B))
                        print("AMStage-Opt hyper-parms", i + 1, batch_idx, out.detach().numpy(),"est is.{}".format(fitting_obj.c.detach()), fitting_obj.base_kernel_parm1.detach(),fitting_obj.base_kernel_parm2.detach())

                # Step 2. minimizing the objective w.r.t. theta and c when fixing kernel B.
                for k in range(1):
                    optimizer.zero_grad()
                    out = fitting_obj(batch_sample_indices = batch_indices, regularizer_const=  regularizer_const, regularizer_const_FB =  regularizer_const_FB, ifopt_hyper_parms = False, ifopt_model_parms = True)
                    out.backward()
                    optimizer.step()

                    if verbose:
                        print("Current Lr is {}.".format(optimizer.param_groups[0]['lr']))
                        print("AMStage-Opt model parms", i + 1, batch_idx, fitting_obj.c.requires_grad,fitting_obj.theta.requires_grad, fitting_obj.base_kernel_parm1.requires_grad,fitting_obj.base_kernel_parm2.requires_grad,  fitting_obj.B.requires_grad)      # print("AMStage-Opt model parms", i + 1, batch_idx, fitting_obj.c.grad,fitting_obj.base_kernel_parm1.grad, fitting_obj.base_kernel_parm2.grad)
                        print("AMStage-Opt model parms", i + 1, batch_idx, out.detach().numpy(),"est is.{}".format(fitting_obj.c.detach()), fitting_obj.base_kernel_parm1.detach(),fitting_obj.base_kernel_parm2.detach())

            # Lr step
            scheduler.step()

            # Random shuffle
            np.random.shuffle(train_indices)
            saved_BQ_est[i,] = fitting_obj.c
            saved_loss[i] = out

        self.fitting_obj = fitting_obj
        self.saved_BQ_est = saved_BQ_est
        self.saved_loss = saved_loss



