from Borehole_Example.src_ProdKernel_vvCV_MD1P.utils_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.score_funcs_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.stein_operators_borehole import *

##=======================================================================================================================================
## 1. For Scalar-Valued CV, and in particular for Borehole functions
##=======================================================================================================================================

class penalized_ls_objective_scalarvaluedfunc_borehole(torch.nn.Module):
    def __init__(self, optim_base_kernel_parms, prior_kernel, base_kernel, batch_size, X_whole_tr, Y_whole_tr, score_tensor_X_whole_tr):
        """
        Recall that we assume a zero mean function as prior mean function
        :param optim_base_kernel_parms: optimized kernel parameters
        :param prior_kernel:  a class, i.e. stein here
        :param base_kernel:   a class
        :param batch_size:    1d value, e.g, 5
        :param X_whole_tr:    2d tensor, m * d
        :param Y_whole_tr:    2d tensor, m * 1
        :param score_tensor_X_whole_tr: 2d tensor, m * d
        """
        super(penalized_ls_objective_scalarvaluedfunc_borehole, self).__init__()

        assert optim_base_kernel_parms.dim() == 2, "optim_base_kernel_parms should be a 2d tensor, d*1"

        # Model parameters
        self.theta = torch.nn.Parameter(torch.zeros(X_whole_tr.size()[0], 1, dtype=torch.float, requires_grad=True))   # 2d tensor, m * 1
        self.c = torch.nn.Parameter(torch.ones(1, dtype=torch.float32, requires_grad=True) * Y_whole_tr.mean())       # 1d tensor, 1

        #
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.batch_size = batch_size
        self.X_whole_tr = X_whole_tr
        self.Y_whole_tr = Y_whole_tr
        self.score_tensor_X_whole_tr = score_tensor_X_whole_tr

        # Get optimized kernel parameters
        self.base_kernel_parm1 = optim_base_kernel_parms[0].detach()
        self.base_kernel_parm2 = optim_base_kernel_parms[1].detach()
        self.base_kernel_parm3 = optim_base_kernel_parms[2].detach()
        self.base_kernel_parm4 = optim_base_kernel_parms[3].detach()
        self.base_kernel_parm5 = optim_base_kernel_parms[4].detach()
        self.base_kernel_parm6 = optim_base_kernel_parms[5].detach()
        self.base_kernel_parm7 = optim_base_kernel_parms[6].detach()
        self.base_kernel_parm8 = optim_base_kernel_parms[7].detach()

        # Pre-compute the k_0(X,X) for this single dataset X
        kernel_obj = prior_kernel(self.base_kernel)  # instantialized the class

        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2
        kernel_obj.base_kernel_parm3 = self.base_kernel_parm3
        kernel_obj.base_kernel_parm4 = self.base_kernel_parm4
        kernel_obj.base_kernel_parm5 = self.base_kernel_parm5
        kernel_obj.base_kernel_parm6 = self.base_kernel_parm6
        kernel_obj.base_kernel_parm7 = self.base_kernel_parm7
        kernel_obj.base_kernel_parm8 = self.base_kernel_parm8

        k_X_X = kernel_obj.cal_stein_base_kernel(self.X_whole_tr, self.X_whole_tr, self.score_tensor_X_whole_tr, self.score_tensor_X_whole_tr)
        self.k_X_X = k_X_X



    def forward(self, batch_sample_indices, regularizer_const):
        """
        :param X_batch: a mini-batch of samples, b * d, where b is batch size and d is dimension of each data point.
        :param Y_batch: a mini-batch of corresponding function/response values.
        :param prior_kernel: A kernel class, eg. Stein
        :param distribution: the pytorch distribution object for the distribution of x -- \Pi
        :param regularizer_const:
        :return:  a value of the objective function which will be used to update some parameters.
        """

        X_batch = self.X_whole_tr[batch_sample_indices,]   # 2d tensor, b * d, contain b samples
        Y_batch = self.Y_whole_tr[batch_sample_indices,]   # 2d tensor, b * 1, contain b samples

        assert X_batch.size()[0] == self.batch_size

        k_Xb_X = self.k_X_X[batch_sample_indices, ]    # 2d tensor, b * m

        theta_k_X_Xb = self.theta.t() @ k_Xb_X.t()
        y_offset_c = Y_batch - self.c
        if y_offset_c.dim() == 1:
            y_offset_c.unsqueeze(dim=1)  # set to be a column vector

        scalable_objective = (theta_k_X_Xb @ theta_k_X_Xb.t() - 2 * (theta_k_X_Xb @ y_offset_c) + y_offset_c.pow(2).sum()) / self.batch_size + regularizer_const * (self.theta.t() @ self.theta)


        return scalable_objective.squeeze()





class SV_CV_scalarvaluedfuncs_model_borehole(object):

    def __init__(self, sv_cv_objective, prior_kernel, base_kernel, X_train, Y_train, score_tensor):
        """
        :param sv_vv_objective: an objective class, e.g. penalized_ls_objective
        :param prior_kernel: a kernel class, here is a stein kernel class.
        :param base_kernel: a kernel class
        :param X_train:  2d tensor, m * d
        :param Y_train:  2d tensor, m *  1
        :param score_tensor:  2d tensor, m * d
        """
        self.sv_cv_objective = sv_cv_objective
        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel
        self.X_train = X_train
        self.Y_train = Y_train
        self.score_tensor = score_tensor


    # Tune kernel hyper-parameters w.r.t. sum of log marginal likelihood
    def do_tune_kernelparams_negmllk(self, batch_size_tune, flag_if_use_medianheuristic=False, beta_cstkernel=1, lr=0.1, epochs=100, verbose=True):
        tune_kernelparams_negmllk_obj = TuneKernelParams_mllk_MRI_singledat_borehole(self.prior_kernel, self.base_kernel,  self.X_train, self.Y_train, self.score_tensor)
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


    def do_optimize_sv_CV(self, regularizer_const, batch_size = 10 , lr=0.2, epochs=200, verbose=True):

        fitting_obj = self.sv_cv_objective(self.optim_base_kernel_parms, self.prior_kernel,self.base_kernel, batch_size, self.X_train, self.Y_train, self.score_tensor)
        optimizer = torch.optim.Adam(fitting_obj.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        saved_BQ_est = torch.zeros(epochs)
        saved_loss = torch.zeros(epochs)
        m = self.X_train.size()[0]
        train_indices = list(range(m))

        for i in range(epochs):

            batches_generator = self.chunks(train_indices, batch_size)  # this creates a generator

            if i >= 0:
                for batch_idx, batch_indices in enumerate(batches_generator):
                    optimizer.zero_grad()
                    out = fitting_obj(batch_indices, regularizer_const)
                    out.backward()
                    optimizer.step()
                    if verbose:
                        print("Regular training stage")
                        print("AMStage-Opt model parms", i + 1, batch_idx, fitting_obj.c.requires_grad, fitting_obj.theta.requires_grad, fitting_obj.base_kernel_parm1.requires_grad,fitting_obj.base_kernel_parm2.requires_grad)
                        print("AMStage-Opt model parms", i + 1, batch_idx, fitting_obj.c.grad, fitting_obj.base_kernel_parm1.grad, fitting_obj.base_kernel_parm2.grad)
                        print("AMStage-Opt model parms", i + 1, batch_idx, out.detach().numpy(),"est is.{}".format(fitting_obj.c.detach()), fitting_obj.base_kernel_parm1.detach(), fitting_obj.base_kernel_parm2.detach())


            scheduler.step()
            np.random.shuffle(train_indices)

            saved_BQ_est[i] = fitting_obj.c
            saved_loss[i] = out

        self.fitting_obj = fitting_obj  # store it and use its optimized hyper-params in the following function.
        self.saved_BQ_est = saved_BQ_est
        self.saved_loss = saved_loss


    def do_integration(self, X_train, Y_train):
        #  A biased estimate for the integral, which has excellent MSE, as suggested by JRSSB-CF
        return self.fitting_obj.c



    def do_integration_ControlVariates(self, X_test, Y_test, score_X_test_tensor):
        # This is for control variates, an unbiased estimate for the integral.
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)

        # Feed the optimzed kernel params
        kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
        kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]
        kernel_obj.base_kernel_parm3 = self.optim_base_kernel_parms[2]
        kernel_obj.base_kernel_parm4 = self.optim_base_kernel_parms[3]
        kernel_obj.base_kernel_parm5 = self.optim_base_kernel_parms[4]
        kernel_obj.base_kernel_parm6 = self.optim_base_kernel_parms[5]
        kernel_obj.base_kernel_parm7 = self.optim_base_kernel_parms[6]
        kernel_obj.base_kernel_parm8 = self.optim_base_kernel_parms[7]

        k_ZX = kernel_obj.cal_stein_base_kernel(X_test, self.X_train, score_X_test_tensor, self.score_tensor)

        if len(Y_test.size()) == 1:
            Y_test = Y_test.unsqueeze(dim=1)  #
        cv_est = (Y_test - k_ZX @ self.fitting_obj.theta).mean().squeeze()

        cv_est_var = torch.var(Y_test - k_ZX @ self.fitting_obj.theta) / Y_test.size()[0]

        return cv_est, cv_est_var


    def do_closed_form_est_for_simpliedCF(self):
        # Simplified CF estimate
        # Instantialized the class
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)

        # Feed the optimzed kernel params
        kernel_obj.base_kernel_parm1 = self.optim_base_kernel_parms[0]
        kernel_obj.base_kernel_parm2 = self.optim_base_kernel_parms[1]
        kernel_obj.base_kernel_parm3 = self.optim_base_kernel_parms[2]
        kernel_obj.base_kernel_parm4 = self.optim_base_kernel_parms[3]
        kernel_obj.base_kernel_parm5 = self.optim_base_kernel_parms[4]
        kernel_obj.base_kernel_parm6 = self.optim_base_kernel_parms[5]
        kernel_obj.base_kernel_parm7 = self.optim_base_kernel_parms[6]
        kernel_obj.base_kernel_parm8 = self.optim_base_kernel_parms[7]

        k_XX = kernel_obj.cal_stein_base_kernel(self.X_train, self.X_train, self.score_tensor, self.score_tensor)

        m = self.X_train.size()[0]
        o  = (torch.ones(1, m )  @ (k_XX + 0.001 * torch.eye(m)).inverse() @ self.Y_train )/( torch.ones(1, m)  @ (k_XX + 0.001 * torch.eye(m)).inverse() @ torch.ones( self.X_train.size()[0], 1 )  )
        return o

