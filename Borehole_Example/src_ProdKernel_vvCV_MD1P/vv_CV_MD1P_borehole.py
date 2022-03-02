from Borehole_Example.src_ProdKernel_vvCV_MD1P.utils_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.score_funcs_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.stein_operators_borehole import *


##=======================================================================================================================================
## 3. vvCVs for Borehole example
##=======================================================================================================================================
class penalized_ls_objective_vectorvaluedfunc_borehole(torch.nn.Module):
    def __init__(self, optim_base_kernel_parms, prior_kernel, base_kernel, batch_size, Xs_tensor, Ys_tensor, scores_Tensor):
        """
        :param optim_base_kernel_parms:
        :param prior_kernel: a class, e.g. stein
        :param base_kernel:  a class, e.g base_kernel_2
        :param batch_size:   a number, e.g. 5
        :param Xs_tensor:    3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param Ys_tensor:    3d tensor, T * m * 1 when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param scores_Tensor: 3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        """
        super(penalized_ls_objective_vectorvaluedfunc_borehole, self).__init__()

        assert optim_base_kernel_parms.dim() == 2, "optim_base_kernel_parms should be a 2d tensor, d*1"

        self.T = Xs_tensor.size()[0]
        self.m = Xs_tensor.size()[1]
        self.d = Xs_tensor.size()[2]
        self.total_no_points = self.T * self.m

        Y_means = Ys_tensor.mean(dim=1)  # 2d tensor; T * 1
        self.Xs_flattened = Xs_tensor.reshape(self.total_no_points, -1)     # 2d tensor;  self.total_no_points * d; all datasets are stacked along rows
        self.scores_Tensor_flattened = scores_Tensor.reshape(self.total_no_points, -1)   # 2d tensor;  self.total_no_points * d; all datasets are stacked along rows

        # Initializing raw params for constructing B
        _ini = torch.log(torch.Tensor([0.0032]))  #
        self.paramsB = torch.nn.Parameter(torch.zeros(self.T, self.T, dtype=torch.float32, requires_grad=True) + torch.diag(_ini.repeat(self.T)))  # self._registered_B = None


        # Initializing model parameters (theta, c)
        self.theta = torch.nn.Parameter(torch.zeros(self.total_no_points, self.T, dtype=torch.float, requires_grad=True))
        self.c =  torch.nn.Parameter(torch.ones(self.T, 1, dtype=torch.float32, requires_grad=True) * Y_means) # column means are MC estimate for each integral, a p * 1 tensor

        #
        self.prior_kernel =  prior_kernel
        self.base_kernel = base_kernel
        self.batch_size = batch_size
        self.Xs_tensor = Xs_tensor
        self.Ys_tensor = Ys_tensor
        self.scores_Tensor = scores_Tensor

        # Get optimized kernel parameters
        self.base_kernel_parm1 = optim_base_kernel_parms[0].detach()
        self.base_kernel_parm2 = optim_base_kernel_parms[1].detach()
        self.base_kernel_parm3 = optim_base_kernel_parms[2].detach()
        self.base_kernel_parm4 = optim_base_kernel_parms[3].detach()
        self.base_kernel_parm5 = optim_base_kernel_parms[4].detach()
        self.base_kernel_parm6 = optim_base_kernel_parms[5].detach()
        self.base_kernel_parm7 = optim_base_kernel_parms[6].detach()
        self.base_kernel_parm8 = optim_base_kernel_parms[7].detach()



        # Instantialized the class
        kernel_obj = self.prior_kernel(self.base_kernel)

        # Feed the optimal kernel params
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2
        kernel_obj.base_kernel_parm3 = self.base_kernel_parm3
        kernel_obj.base_kernel_parm4 = self.base_kernel_parm4
        kernel_obj.base_kernel_parm5 = self.base_kernel_parm5
        kernel_obj.base_kernel_parm6 = self.base_kernel_parm6
        kernel_obj.base_kernel_parm7 = self.base_kernel_parm7
        kernel_obj.base_kernel_parm8 = self.base_kernel_parm8

        # Compute kernel
        self.k_Xs_Xs_tensor_flattened = kernel_obj.cal_stein_base_kernel(self.Xs_flattened , self.Xs_flattened , self.scores_Tensor_flattened, self.scores_Tensor_flattened)




    def forward(self, batch_sample_indices, regularizer_const, regularizer_const_FB, ifopt_hyper_parms, ifopt_model_parms):

        assert len(batch_sample_indices)  ==  self.batch_size

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
        self.diag_paramsB = torch.exp(torch.diag(self.LowerTri_paramsB)) # ensure diagonal elements > 0 (recall LL^T decomposition of psd matrix)
        mask = torch.diag(torch.ones_like(self.diag_paramsB))  # therefore, if the diagonal elements of self.paramsB are zero intially, then self.B will have diagonal elements equal to 1.
        self.LowerTriAdj_paramsB = mask * torch.diag(self.diag_paramsB) + (1. - mask) * self.LowerTri_paramsB
        self.B = self.LowerTriAdj_paramsB @ self.LowerTriAdj_paramsB.t()



        Xs_batch = self.Xs_tensor[:, batch_sample_indices, :].reshape(self.T * self.batch_size, -1)  # 2d tensor , (Tb) * d
        Ys_batch = self.Ys_tensor[:, batch_sample_indices, :].reshape(self.T * self.batch_size, -1)  # 2d tensor , (Tb) * 1

        preds = torch.zeros(self.T, self.batch_size)
        for idx in range(self.T):
            cur_indices = torch.tensor(batch_sample_indices) + idx * self.m  #  torch.tensor force a long type which can be used as indices
            cur_indices = torch.tensor(cur_indices)
            preds[idx,] =   self.k_Xs_Xs_tensor_flattened[cur_indices,] @ self.theta @ self.B[idx, ].t() + self.c[idx, ]

        preds = preds.reshape(self.T * self.batch_size, -1)
        assert preds.size() == Ys_batch.size()
        obj_batch = (1. / self.batch_size) * (Ys_batch - preds).pow(2).sum()

        penlty_on_B = torch.norm(self.B, p='fro').pow(2)
        scalable_objective = obj_batch + regularizer_const_FB * penlty_on_B + regularizer_const * torch.norm(self.theta, p='fro').pow(2)


        return scalable_objective.squeeze()





class VV_CV_vectorvaluedfuncs_model_borehole(object):
    def __init__(self, vv_cv_objective, prior_kernel, base_kernel, Xs_tensor, Ys_tensor, scores_Tensor):
        """
        :param vv_cv_objective:  an objective class, e.g. penalized_ls_objective
        :param prior_kernel:  a class, e.g. stein
        :param base_kernel:   a class, e.g. base_kernel_2
        :param Xs_tensor:    3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param Ys_tensor:    3d tensor, T * m * 1 when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param scores_Tensor: 3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        """

        self.vv_cv_objective = vv_cv_objective
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.Xs_tensor = Xs_tensor
        self.Ys_tensor = Ys_tensor
        self.scores_Tensor = scores_Tensor



    # Tune kernel hyper-parameters w.r.t. sum of log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_multidats_borehole( self.prior_kernel, self.base_kernel, self.Xs_tensor, self.Ys_tensor, self.scores_Tensor)
        tune_kernelparams_negmllk_obj.do_optimize_logmll(batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr, epochs, verbose)

        optim_base_kernel_parms = torch.Tensor([tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm1, \
                                                tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm2, \
                                                tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm3, \
                                                tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm4, \
                                                tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm5, \
                                                tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm6, \
                                                tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm7, \
                                                tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm8])

        self.optim_base_kernel_parms =optim_base_kernel_parms.detach().unsqueeze(dim=1)
        return optim_base_kernel_parms.detach()


    # Split an iterable of items into batches
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

        optimizer = torch.optim.Adam([
                          {'params': fitting_obj.theta},
                          {'params': fitting_obj.c},
                          {'params': fitting_obj.paramsB, 'lr': 2e-5}
                    ], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        saved_BQ_est = torch.zeros(epochs, fitting_obj.T,  1)
        saved_loss  = torch.zeros(epochs, 1)
        m = self.Xs_tensor.size()[1]
        train_indices = list(range(m))
        for i in range(epochs):

            batches_generator = self.chunks(train_indices, batch_size)  # this creates a generator

            for batch_idx, batch_indices in enumerate(batches_generator):

                # Step 1. minimizing the objective w.r.t. hyperparameters when fixing theta and c.
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

                # Step 2. minimizing the objective w.r.t. theta and c when fixing kernel hyperparameters.
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
            # print(i, fitting_obj.c)
            saved_BQ_est[i,] = fitting_obj.c
            saved_loss[i] = out

        self.fitting_obj = fitting_obj  # store it and use its optimized hyper-params in the following function.
        self.saved_BQ_est = saved_BQ_est
        self.saved_loss = saved_loss


    def do_closed_form_est_for_simpliedCF(self):
        T = self.Xs_tensor.size()[0]
        m = self.Xs_tensor.size()[1]
        o = []
        for idx in range(T):
            Y_l = self.Ys_tensor[idx, ]              # 2d tensor, m * 1
            X_l = self.Xs_tensor[idx, ]              # 2d tensor, m * d
            score_X_l = self.scores_Tensor[idx, ]    # 2d tensor, m * d

            # Instantialized the class
            kernel_obj = self.prior_kernel(self.base_kernel)

            # Feed the optimzed kernel params
            kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
            kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]
            kernel_obj.base_kernel_parm3 = self.optim_base_kernel_parms[2]
            kernel_obj.base_kernel_parm4 = self.optim_base_kernel_parms[3]
            kernel_obj.base_kernel_parm5 = self.optim_base_kernel_parms[4]
            kernel_obj.base_kernel_parm6 = self.optim_base_kernel_parms[5]
            kernel_obj.base_kernel_parm7 = self.optim_base_kernel_parms[6]
            kernel_obj.base_kernel_parm8 = self.optim_base_kernel_parms[7]

            k_Xl_Xl = kernel_obj.cal_stein_base_kernel(X_l, X_l, score_X_l, score_X_l)

            cv_est_l =  (torch.ones(1, m) @ (k_Xl_Xl + 0.001 * torch.eye(m)).inverse() @ Y_l) / (torch.ones(1, m) @ (k_Xl_Xl + 0.001 * torch.eye(m)).inverse() @ torch.ones(m, 1))
            o.append(cv_est_l.detach().numpy())

        return o


