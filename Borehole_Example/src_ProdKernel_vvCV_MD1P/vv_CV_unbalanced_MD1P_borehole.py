from Borehole_Example.src_ProdKernel_vvCV_MD1P.utils_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.score_funcs_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.stein_operators_borehole import *



class penalized_ls_objective_vectorvaluedfunc_unbalanced_borehole(torch.nn.Module):
    def __init__(self, optim_base_kernel_parms, prior_kernel, base_kernel, batch_size, Xs_tuple, Ys_tuple, scores_tuple):
        """
        Unbalanced case: we assume the first component is the dataset of the high-fidelity case
        -------
        :param optim_base_kernel_parms:
        :param prior_kernel:  a class, e.g. stein
        :param base_kernel:   a class, e.g base_kernel_2
        :param batch_size:    a number, e.g. 5 -- for unbalanced cases, it is a base batch size and the sample size will be proportion to the size of the corresponding dataset
        :param Xs_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        :param Ys_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, 1] , ... , [mT, 1])
        :param scores_tuple:  a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        """
        super(penalized_ls_objective_vectorvaluedfunc_unbalanced_borehole, self).__init__()

        self.T = len(Xs_tuple)
        _m_list = []
        Y_means = torch.zeros(self.T, 1)  # 2d tensor; T * 1
        _Xs_flattened = torch.Tensor()    # 2d tensor; (\sum_j m_j) * d
        _Ys_flattened = torch.Tensor()    # 2d tensor; (\sum_j m_j) * 1
        _scores_Tensor_flattened = torch.Tensor()
        for i in range(self.T):
            _m_list.append(Xs_tuple[i].size()[0])
            Y_means[i,] = Ys_tuple[i].mean()
            _Xs_flattened = torch.cat((_Xs_flattened, Xs_tuple[i]), dim=0)
            _Ys_flattened = torch.cat((_Ys_flattened, Ys_tuple[i]), dim=0)
            _scores_Tensor_flattened = torch.cat((_scores_Tensor_flattened, scores_tuple[i]), dim=0)



        self.m_list = torch.tensor(_m_list)     # transfer to torch.long type; 1d tensor longtype
        self.m_base = self.m_list[-1]

        self.d = Xs_tuple[0].size(1)           # as we assume all datapoints from same distriburion and therefore they have identical dimension
        self.total_no_points = self.m_list.sum()                   # \sum_j m_j
        self.Xs_flattened = _Xs_flattened                          # 2d tensor; (\sum_j m_j) * d
        self.Ys_flattened = _Ys_flattened
        self.scores_Tensor_flattened = _scores_Tensor_flattened    # 2d tensor; (\sum_j m_j) * d


        # Initializing raw params for constructing B
        _ini = torch.log(torch.Tensor([0.0032]))  # this means self.B is initialized at diag(1e-5, ...1e-5)
        self.paramsB = torch.nn.Parameter(torch.zeros(self.T, self.T, dtype=torch.float32, requires_grad=True) + torch.diag(_ini.repeat(self.T)))



        # Initializing model parameters (theta, c)
        self.theta = torch.nn.Parameter(torch.zeros(self.total_no_points, self.T, dtype=torch.float, requires_grad=True))
        self.c =  torch.nn.Parameter(torch.ones(self.T, 1, dtype=torch.float32, requires_grad=True) * Y_means)

        #
        self.prior_kernel =  prior_kernel
        self.base_kernel = base_kernel
        self.batch_size = batch_size
        self.Xs_tuple = Xs_tuple
        self.Ys_tuple = Ys_tuple
        self.scores_tuple = scores_tuple


        # Get optimized kernel parameters
        self.base_kernel_parm1 = optim_base_kernel_parms[0].detach()
        self.base_kernel_parm2 = optim_base_kernel_parms[1].detach()
        self.base_kernel_parm3 = optim_base_kernel_parms[2].detach()
        self.base_kernel_parm4 = optim_base_kernel_parms[3].detach()
        self.base_kernel_parm5 = optim_base_kernel_parms[4].detach()
        self.base_kernel_parm6 = optim_base_kernel_parms[5].detach()
        self.base_kernel_parm7 = optim_base_kernel_parms[6].detach()
        self.base_kernel_parm8 = optim_base_kernel_parms[7].detach()


        #
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




    def forward(self, batch_sample_indices_dict, regularizer_const, regularizer_const_FB, ifopt_hyper_parms, ifopt_model_parms):
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


        accum_ms = 0
        obj_batch = 0.
        for idx in range(self.T):
            cur_indices = torch.tensor(batch_sample_indices_dict[idx]) + accum_ms #  torch.tensor force a long type which can be used as indices
            cur_preds =   self.k_Xs_Xs_tensor_flattened[cur_indices,] @ self.theta @ self.B[idx, ].t() + self.c[idx, ]
            cur_Y_batch = self.Ys_flattened[cur_indices, ]
            cur_Y_batch = cur_Y_batch.squeeze()

            cur_preds = cur_preds.squeeze()
            assert cur_preds.size() == cur_Y_batch.size(),"cur_preds size is {}; cur_Y_batch size is {}".format(cur_preds.size() ,  cur_Y_batch.size())

            obj_batch = obj_batch + (self.m_base/(self.batch_size * (self.m_list[idx]/self.m_base))) * (cur_Y_batch - cur_preds).pow(2).sum()

            accum_ms = accum_ms + self.m_list[idx]

        assert accum_ms == self.m_list.sum(), "accum_ms is {} and self.m_list is {} ".format(accum_ms, self.m_list)



        penlty_on_B = torch.norm(self.B, p='fro').pow(2)
        scalable_objective = obj_batch + regularizer_const_FB * penlty_on_B + regularizer_const * torch.norm(self.theta, p='fro').pow(2)

        return scalable_objective.squeeze()








class VV_CV_vectorvaluedfuncs_model_unbalanced_borehole(object):
    def __init__(self, vv_cv_objective, prior_kernel, base_kernel, Xs_tuple, Ys_tuple, scores_tuple):
        """
        :param vv_cv_objective:  an objective class, e.g. penalized_ls_objective
        :param prior_kernel:  a class, e.g. stein
        :param base_kernel:   a class, e.g. base_kernel_2
        :param Xs_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        :param Ys_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, 1] , ... , [mT, 1])
        :param scores_tuple:  a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        """

        self.vv_cv_objective = vv_cv_objective
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.Xs_tuple = Xs_tuple
        self.Ys_tuple = Ys_tuple
        self.scores_tuple = scores_tuple


    # Tune kernel hyper-parameters w.r.t. sum of log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_multidats_unbalanced_borehole(self.prior_kernel, self.base_kernel, self.Xs_tuple, self.Ys_tuple, self.scores_tuple)
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
        fitting_obj = self.vv_cv_objective(self.optim_base_kernel_parms, self.prior_kernel, self.base_kernel, batch_size,  self.Xs_tuple, self.Ys_tuple, self.scores_tuple)
        optimizer = torch.optim.Adam([
                          {'params': fitting_obj.theta},
                          {'params': fitting_obj.c},
                          {'params': fitting_obj.paramsB, 'lr': 2e-5}
                    ], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        saved_BQ_est = torch.zeros(epochs, fitting_obj.T,  1)
        saved_loss  = torch.zeros(epochs, 1)
        m_base = fitting_obj.m_base
        train_indices = list(range(m_base))
        for i in range(epochs):

            batches_generator = self.chunks(train_indices, batch_size)  #

            for batch_idx, batch_indices in enumerate(batches_generator):
                # Use batch_indices to create a corresponding correct tuple of batch_indices
                batch_indices_dict = {}
                for cr in range(fitting_obj.T):
                    start_pos = batch_idx * batch_size * (fitting_obj.m_list[cr] / fitting_obj.m_base)
                    end_pos = (batch_idx + 1) * batch_size * (fitting_obj.m_list[cr] / fitting_obj.m_base) - 1
                    batch_indices_dict[cr] = list(range(int(start_pos.item()), int(end_pos.item()) + 1))


                # Step 1. minimizing the objective w.r.t. hyperparameters when fixing theta and c.
                for k in range(1):
                    optimizer.zero_grad()
                    out = fitting_obj( batch_sample_indices_dict = batch_indices_dict, regularizer_const = regularizer_const, regularizer_const_FB = regularizer_const_FB, ifopt_hyper_parms = True, ifopt_model_parms = False)
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
                    out = fitting_obj(batch_sample_indices_dict = batch_indices_dict, regularizer_const=  regularizer_const, regularizer_const_FB =  regularizer_const_FB, ifopt_hyper_parms = False, ifopt_model_parms = True)
                    out.backward()
                    optimizer.step()

                    if verbose:
                        print("Current Lr is {}.".format(optimizer.param_groups[0]['lr']))
                        print("AMStage-Opt model parms", i + 1, batch_idx, fitting_obj.c.requires_grad,fitting_obj.theta.requires_grad, fitting_obj.base_kernel_parm1.requires_grad,fitting_obj.base_kernel_parm2.requires_grad,  fitting_obj.B.requires_grad)
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



