from src.src_vvCV_MD1P.utils import *
from src.src_vvCV_MD1P.score_funcs import *




class penalized_ls_objective_vectorvaluedfunc_unbalanced_fixB(torch.nn.Module):
    def __init__(self, B, optim_base_kernel_parms, prior_kernel, base_kernel, batch_size, Xs_tuple, Ys_tuple, scores_tuple):
        """
        Recall that we assume a zero mean function as prior mean function
        Unbalanced case: we assume the first component of the dataset-tuple is the dataset of high-fidelity case
        -------
        :param optim_base_kernel_parms:
        :param prior_kernel:  a class, e.g. stein
        :param base_kernel:   a class, e.g base_kernel_2
        :param batch_size:    a number, e.g. 5  -- for unbalanced cases, it is a base batch size and the sample size will be proportion to the size of the corresponding dataset
        :param Xs_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        :param Ys_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, 1] , ... , [mT, 1])
        :param scores_tuple:  a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        """
        super(penalized_ls_objective_vectorvaluedfunc_unbalanced_fixB, self).__init__()

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
        self.m_base = self.m_list[0]
        self.d = Xs_tuple[0].size(1)           # as we assume all datapoints from same distriburion and therefore they have identical dimension
        self.total_no_points = self.m_list.sum()                   # \sum_j m_j
        self.Xs_flattened = _Xs_flattened                          # 2d tensor; (\sum_j m_j) * d
        self.Ys_flattened = _Ys_flattened
        self.scores_Tensor_flattened = _scores_Tensor_flattened    # 2d tensor; (\sum_j m_j) * d


        # Set B
        self.B = B

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

        #
        self.base_kernel_parm1 = optim_base_kernel_parms[0].detach()
        self.base_kernel_parm2 = optim_base_kernel_parms[1].detach()


        #
        kernel_obj = self.prior_kernel(self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2
        self.k_Xs_Xs_tensor_flattened = kernel_obj.cal_stein_base_kernel(self.Xs_flattened , self.Xs_flattened , self.scores_Tensor_flattened, self.scores_Tensor_flattened)




    def forward(self, batch_sample_indices_dict, regularizer_const):
        """
        :param batch_sample_indices_tuple: a tuple of batch_sample_indices, i.e. ([bsi_1, bsi_2, ..., bsi_T])
        :param regularizer_const:
        :param regularizer_const_FB:
        :return:
        """
        accum_ms = 0
        obj_batch = 0.
        for idx in range(self.T):
            cur_indices = torch.tensor(batch_sample_indices_dict[idx]) + accum_ms #  torch.tensor force a long type which can be used as indices
            cur_preds =   self.k_Xs_Xs_tensor_flattened[cur_indices,] @ self.theta @ self.B[idx, ].t() + self.c[idx, ]
            cur_Y_batch = self.Ys_flattened[cur_indices, ]

            cur_preds = cur_preds.unsqueeze(dim=1)
            assert cur_preds.size() == cur_Y_batch.size(), "cur_preds size is {}; cur_Y_batch size is {}".format(cur_preds.size(), cur_Y_batch.size())
            obj_batch = obj_batch + (1. / (self.batch_size * (self.m_list[idx] / self.m_base))) * (cur_Y_batch - cur_preds).pow(2).sum()

            accum_ms = accum_ms + self.m_list[idx]


        scalable_objective = obj_batch + regularizer_const * torch.norm(self.theta, p='fro').pow(2) #  torch.trace(self.theta.t() @ self.theta)

        return scalable_objective.squeeze()





class VV_CV_vectorvaluedfuncs_model_unbalanced_fixB(object):
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

        self.T = len(Xs_tuple)
        self._B = torch.eye(self.T)

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, x):
        assert x.size() == torch.ones(self.T, self.T).size(), "The B you input should be T-by-T."
        assert torch.det(x) >= 0, "The B you input should be a PSD matrix."
        self._B = x


    # Tune kernel hyper-parameters w.r.t. sum of log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_multidats_unbalanced(self.prior_kernel, self.base_kernel, self.Xs_tuple, self.Ys_tuple, self.scores_tuple)
        tune_kernelparams_negmllk_obj.do_optimize_logmll(batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr, epochs, verbose)
        optim_base_kernel_parms = torch.Tensor([tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm1, tune_kernelparams_negmllk_obj.neg_mll.base_kernel_parm2])
        self.optim_base_kernel_parms =optim_base_kernel_parms.detach()
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
            yield ls[i:i + batch_size]  # note that the 'i+batch_size'th item is not included; thus we have 'batch_size' of samples



    # Optimize B and (Theta, c) via alternating minimization
    def do_optimize_vv_CV(self, regularizer_const = 0.01, batch_size = 10, lr=0.2, epochs=200, verbose=True):

        fitting_obj = self.vv_cv_objective(self.B, self.optim_base_kernel_parms, self.prior_kernel, self.base_kernel, batch_size,  self.Xs_tuple, self.Ys_tuple, self.scores_tuple)
        optimizer = torch.optim.Adam(fitting_obj.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        saved_BQ_est = torch.zeros(epochs, fitting_obj.T,  1)
        saved_loss  = torch.zeros(epochs, 1)
        m_base = fitting_obj.m_base
        train_indices = list(range(m_base))
        for i in range(epochs):

            batches_generator = self.chunks(train_indices, batch_size)  # this creates a generator

            for batch_idx, batch_indices in enumerate(batches_generator):
                # Use batch_indices to create a corresponding correct tuple of batch_indices
                batch_indices_dict = {}
                for cr in range(fitting_obj.T):
                    start_pos = batch_idx * batch_size * (fitting_obj.m_list[cr] / fitting_obj.m_base)
                    end_pos = (batch_idx + 1) * batch_size * (fitting_obj.m_list[cr] / fitting_obj.m_base) - 1
                    batch_indices_dict[cr] = list(range(int(start_pos.item()), int(end_pos.item()) + 1))

                optimizer.zero_grad()
                out = fitting_obj( batch_sample_indices_dict = batch_indices_dict, regularizer_const=  regularizer_const)
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

        self.fitting_obj = fitting_obj  # store it and use its optimized hyper-params in the following function.
        self.saved_BQ_est = saved_BQ_est
        self.saved_loss = saved_loss




