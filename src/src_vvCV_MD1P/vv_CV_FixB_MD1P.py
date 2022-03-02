from src.src_vvCV_MD1P.utils import *
from src.src_vvCV_MD1P.score_funcs import *




##=======================================================================================================================================
## 1. vvCVs with B fixed
##=======================================================================================================================================

class penalized_ls_objective_vectorvaluedfunc_fixB(torch.nn.Module):
    def __init__(self, B, optim_base_kernel_parms, prior_kernel, base_kernel, batch_size, Xs_tensor, Ys_tensor, scores_Tensor):
        """
        :param optim_base_kernel_parms:
        :param prior_kernel: a class, e.g. stein
        :param base_kernel:  a class, e.g base_kernel_2
        :param batch_size:   a number, e.g. 5
        :param Xs_tensor:    3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param Ys_tensor:    3d tensor, T * m * 1 when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param scores_Tensor: 3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        """
        super(penalized_ls_objective_vectorvaluedfunc_fixB, self).__init__()

        self.T = Xs_tensor.size()[0]
        self.m = Xs_tensor.size()[1]
        self.d = Xs_tensor.size()[2]
        self.total_no_points = self.T * self.m

        Y_means = Ys_tensor.mean(dim=1)  # 2d tensor; T * 1
        self.Xs_flattened = Xs_tensor.reshape(self.total_no_points, -1)     # 2d tensor;  self.total_no_points * d; all datasets are stacked along rows
        self.scores_Tensor_flattened = scores_Tensor.reshape(self.total_no_points, -1)   # 2d tensor;  self.total_no_points * d; all datasets are stacked along rows

        # Set B
        self.B = B


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

        #
        self.base_kernel_parm1 = optim_base_kernel_parms[0].detach()
        self.base_kernel_parm2 = optim_base_kernel_parms[1].detach()


        kernel_obj = self.prior_kernel(self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2
        self.k_Xs_Xs_tensor_flattened = kernel_obj.cal_stein_base_kernel(self.Xs_flattened , self.Xs_flattened , self.scores_Tensor_flattened, self.scores_Tensor_flattened)




    def forward(self, batch_sample_indices, regularizer_const):

        assert len(batch_sample_indices)  ==  self.batch_size

        Xs_batch = self.Xs_tensor[:, batch_sample_indices, :].reshape(self.T * self.batch_size, -1)  # 2d tensor , (Tb) * d
        Ys_batch = self.Ys_tensor[:, batch_sample_indices, :].reshape(self.T * self.batch_size, -1)  # 2d tensor , (Tb) * 1

        preds = torch.zeros(self.T, self.batch_size)
        for idx in range(self.T):
            cur_indices = torch.tensor(batch_sample_indices) + idx * self.m  #  torch.tensor force a long type which can be used as indices
            cur_indices = torch.tensor(cur_indices)
            preds[idx,] =   self.k_Xs_Xs_tensor_flattened[cur_indices,] @ self.theta @ self.B[idx, ].t() + self.c[idx, ]

        preds = preds.reshape(self.T * self.batch_size, -1)

        obj_batch = (1. / self.batch_size) * (Ys_batch - preds).pow(2).sum()

        scalable_objective = obj_batch + regularizer_const * torch.norm(self.theta, p='fro').pow(2)

        return scalable_objective.squeeze()










class VV_CV_vectorvaluedfuncs_model_fixB(object):
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


        self._B = torch.eye(self.Xs_tensor.size()[0])


    @property
    def B(self):
        return self._B
    @B.setter
    def B(self,x):
        assert x.size() == torch.ones(self.Xs_tensor.size()[0], self.Xs_tensor.size()[0]).size() , "The B you input should be T-by-T."
        assert torch.det(x) >= 0, "The B you input should be a PSD matrix."
        self._B = x




    # Tune kernel hyper-parameters w.r.t. sum of log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_multidats(self.prior_kernel, self.base_kernel, self.Xs_tensor, self.Ys_tensor, self.scores_Tensor)
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
            yield ls[i:i + batch_size]  # ZS: note that the 'i+batch_size'th item is not included; thus we have 'batch_size' of samples



    # Optimize B and (Theta, c) via alternating minimization
    def do_optimize_vv_CV(self, regularizer_const = 0.01,  batch_size = 10, lr=0.2, epochs=200, verbose=True):

        fitting_obj = self.vv_cv_objective(self.B, self.optim_base_kernel_parms, self.prior_kernel, self.base_kernel, batch_size,  self.Xs_tensor, self.Ys_tensor, self.scores_Tensor)
        optimizer = torch.optim.Adam(fitting_obj.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        saved_BQ_est = torch.zeros(epochs, fitting_obj.T,  1)
        saved_loss  = torch.zeros(epochs, 1)
        m = self.Xs_tensor.size()[1]
        train_indices = list(range(m))
        for i in range(epochs):
            batches_generator = self.chunks(train_indices, batch_size)  # ZS: this creates a generator

            for batch_idx, batch_indices in enumerate(batches_generator):
                # Minimizing the objective w.r.t. theta and c when fixing kernel hyperparameters.
                optimizer.zero_grad()
                out = fitting_obj(batch_sample_indices = batch_indices, regularizer_const=  regularizer_const)
                out.backward()
                optimizer.step()

                if verbose:
                    print("Current Lr is {}.".format(optimizer.param_groups[0]['lr']))
                    print("B is {}".format(self.B))
                    print("AMStage-Opt model parms", i + 1, batch_idx, fitting_obj.c.requires_grad,fitting_obj.theta.requires_grad, fitting_obj.base_kernel_parm1.requires_grad,fitting_obj.base_kernel_parm2.requires_grad,  fitting_obj.B.requires_grad)      # print("AMStage-Opt model parms", i + 1, batch_idx, fitting_obj.c.grad,fitting_obj.base_kernel_parm1.grad, fitting_obj.base_kernel_parm2.grad)
                    print("AMStage-Opt model parms", i + 1, batch_idx, out.detach().numpy(),"est is.{}".format(fitting_obj.c.detach()), fitting_obj.base_kernel_parm1.detach(),fitting_obj.base_kernel_parm2.detach())


            scheduler.step()

            # Random shuffle
            np.random.shuffle(train_indices)
            saved_BQ_est[i,] = fitting_obj.c
            saved_loss[i] = out

        self.fitting_obj = fitting_obj
        self.saved_BQ_est = saved_BQ_est
        self.saved_loss = saved_loss



