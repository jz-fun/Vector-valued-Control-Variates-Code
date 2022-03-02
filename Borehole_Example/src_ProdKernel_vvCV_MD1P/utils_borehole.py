from Borehole_Example.src_ProdKernel_vvCV_MD1P.score_funcs_borehole import *


##########################################################################################
# 1. For single dataset: SGD on  log marginal likelihood
##########################################################################################
class negative_log_marginal_lik_MRI_singledat_borehole(torch.nn.Module):
    def __init__(self, prior_kernel, base_kernel, batch_size, X_whole_tr, Y_whole_tr, score_tensor_X_whole_tr, flag_if_use_medianheuristic):
        """
        Recall that we assume a zero mean function as prior mean function
        Once v is optimized, in the SCV-optimization, we can pre-compute the k_0(X,X), for all datasets, i.e. k_0(X_1, X_1) ... k_0(X_1, X_T) ... k_0(X_T, X_T)
        ----
        :param prior_kernel: a class, here should be prod_rbf_kernel_Borehore
        :param base_kernel:  a class, here should be stein_base_kernel_borehole
        :param X_whole_tr:   2d tensor, m * d
        :param Y_whole_tr:   2d tensor, m * 1
        :param score_tensor_X_whole_tr:  2d tensor, m * d
        :param flag_if_use_medianheuristic:
        """
        super(negative_log_marginal_lik_MRI_singledat_borehole, self).__init__()

        if flag_if_use_medianheuristic == False:
            self.base_kernel_parm1_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm2_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm3_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm4_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm5_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm6_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm7_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm8_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))

        self.prior_kernel = prior_kernel  # a class
        self.base_kernel = base_kernel  # a class
        self.X_whole_tr = X_whole_tr
        self.Y_whole_tr = Y_whole_tr
        self.score_tensor_X_whole_tr = score_tensor_X_whole_tr

        self.batch_size = batch_size



    def forward(self, batch_sample_indices, beta_cstkernel=1):

        assert len(batch_sample_indices) == self.batch_size  # ZS: this line is not necessary

        self.base_kernel_parm1 = torch.exp(self.base_kernel_parm1_raw)
        self.base_kernel_parm2 = torch.exp(self.base_kernel_parm2_raw)
        self.base_kernel_parm3 = torch.exp(self.base_kernel_parm3_raw)
        self.base_kernel_parm4 = torch.exp(self.base_kernel_parm4_raw)
        self.base_kernel_parm5 = torch.exp(self.base_kernel_parm5_raw)
        self.base_kernel_parm6 = torch.exp(self.base_kernel_parm6_raw)
        self.base_kernel_parm7 = torch.exp(self.base_kernel_parm7_raw)
        self.base_kernel_parm8 = torch.exp(self.base_kernel_parm8_raw)


        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2
        kernel_obj.base_kernel_parm3 = self.base_kernel_parm3
        kernel_obj.base_kernel_parm4 = self.base_kernel_parm4
        kernel_obj.base_kernel_parm5 = self.base_kernel_parm5
        kernel_obj.base_kernel_parm6 = self.base_kernel_parm6
        kernel_obj.base_kernel_parm7 = self.base_kernel_parm7
        kernel_obj.base_kernel_parm8 = self.base_kernel_parm8

        X_batch =      self.X_whole_tr[batch_sample_indices,:]
        Y_batch =      self.Y_whole_tr[batch_sample_indices,:]
        score_batch =  self.score_tensor_X_whole_tr[batch_sample_indices,:]

        k_XbXb = kernel_obj.cal_stein_base_kernel(X_batch, X_batch, score_batch, score_batch) + beta_cstkernel

        cond_number_threshold = 1e6
        lam = 1e-6

        bad_cond = np.linalg.cond(k_XbXb.detach().numpy()) >= cond_number_threshold
        k_Yb = k_XbXb + lam * torch.eye(X_batch.size()[0])
        while bad_cond:
            lam = 10 * lam
            k_Yb = k_XbXb + lam * torch.eye(X_batch.size()[0])
            bad_cond = np.linalg.cond(k_Yb.detach().numpy()) >= cond_number_threshold
        print("regularization multiplier =", lam)
        k_Yb.to(dtype=torch.float64)

        if Y_batch.dim() == 1:
            Y_batch = Y_batch.unsqueeze(dim=1)

        distrib = torch.distributions.MultivariateNormal(torch.zeros(Y_batch.size()[0]), covariance_matrix=k_Yb)
        log_mll = 0.
        for j in range(self.Y_whole_tr.size()[1]):
            log_mll += distrib.log_prob(Y_batch[:, j].squeeze())

        neg_log_mll = -1. * log_mll
        return neg_log_mll






class TuneKernelParams_mllk_MRI_singledat_borehole(object):

    def __init__(self, prior_kernel, base_kernel, X_train, Y_train, score_tensor):
        """
        :param prior_kernel: a class, stein
        :param base_kernel:  a class
        :param X_train:      2d tensor, m * d
        :param Y_train:      2d tensor, m * d
        :param score_tensor: 2d tensor, m * d
        """

        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel

        self.X_train = X_train
        self.Y_train = Y_train
        self.score_tensor = score_tensor

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



    def do_optimize_logmll(self, batch_size, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.01, epochs=100, verbose=True):
        neg_mll = negative_log_marginal_lik_MRI_singledat_borehole(self.prior_kernel, self.base_kernel, batch_size, self.X_train, self.Y_train, self.score_tensor, flag_if_use_medianheuristic)

        optimizer = torch.optim.Adam(neg_mll.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        m = self.X_train.size()[0]
        train_indices = list(range(m))

        for i in range(epochs):
            batches_generator = self.chunks(train_indices, batch_size)  # this creates a generator
            for batch_idx, batch_indices in enumerate(batches_generator):

                scheduler.step()
                optimizer.zero_grad()
                out = neg_mll(batch_indices, beta_cstkernel)
                out.backward()
                optimizer.step()


                with torch.no_grad():
                    for param in neg_mll.parameters():
                        param.clamp_(min=np.log(1e-3))

                if verbose:
                    print(i + 1, iter, out, neg_mll.base_kernel_parm1.detach(), \
                          neg_mll.base_kernel_parm2.detach(), \
                          neg_mll.base_kernel_parm3.detach(), \
                          neg_mll.base_kernel_parm4.detach(), \
                          neg_mll.base_kernel_parm5.detach(), \
                          neg_mll.base_kernel_parm6.detach(), \
                          neg_mll.base_kernel_parm7.detach(), \
                          neg_mll.base_kernel_parm8.detach())

            # Random shuffle
            np.random.shuffle(train_indices)

        self.neg_mll = neg_mll








##########################################################################################
# 2. For multiple datasets: SGD on sum of log marginal likelihood
##########################################################################################
class negative_log_marginal_lik_MRI_multidats_borehole(torch.nn.Module):
    def __init__(self, prior_kernel, base_kernel, batch_size,  Xs_tensor, Ys_tensor, scores_Tensor, flag_if_use_medianheuristic):
        """
        :Recall that we assume a zero mean function as prior mean function
        :param prior_kernel:   a class
        :param base_kernel:    a class
        :param Xs_tensor:      3d Tensor, T * m * d ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param Ys_tensor:      3d Tensor, T * m * 1 ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param scores_Tensor:  3d Tensor, T * m * d ; grad_x of logp(x) for m points from each of the dataset
        :param flag_if_use_medianheuristic: whether or not to use median heuristic
        """
        super(negative_log_marginal_lik_MRI_multidats_borehole, self).__init__()

        T = Xs_tensor.size()[0]
        m = Xs_tensor.size()[1]
        d = Xs_tensor.size()[2]

        Y_means = Ys_tensor.mean(dim=1)  # 2d tensor; T * 1
        Xs_flattened = Xs_tensor.reshape(T*m, -1) # 2d tensor; (Tm)*d; all datasets are stacked along rows


        # Kernel hyper-parameters
        if flag_if_use_medianheuristic == False:
            self.base_kernel_parm1_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm2_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm3_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm4_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm5_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm6_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm7_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm8_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))

        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel

        self.T = T
        self.m = m
        self.d = d

        self.Xs_tensor = Xs_tensor
        self.Ys_tensor = Ys_tensor
        self.Xs_flattened = Xs_flattened
        self.scores_Tensor = scores_Tensor

        self.batch_size = batch_size



    def forward(self, batch_sample_indices, beta_cstkernel = 1):

        assert len(batch_sample_indices)  ==  self.batch_size

        # Transform the raw params
        self.base_kernel_parm1 = torch.exp(self.base_kernel_parm1_raw)
        self.base_kernel_parm2 = torch.exp(self.base_kernel_parm2_raw)
        self.base_kernel_parm3 = torch.exp(self.base_kernel_parm3_raw)
        self.base_kernel_parm4 = torch.exp(self.base_kernel_parm4_raw)
        self.base_kernel_parm5 = torch.exp(self.base_kernel_parm5_raw)
        self.base_kernel_parm6 = torch.exp(self.base_kernel_parm6_raw)
        self.base_kernel_parm7 = torch.exp(self.base_kernel_parm7_raw)
        self.base_kernel_parm8 = torch.exp(self.base_kernel_parm8_raw)

        # Instantialized the class
        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)

        # Feed the kernel params
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2
        kernel_obj.base_kernel_parm3 = self.base_kernel_parm3
        kernel_obj.base_kernel_parm4 = self.base_kernel_parm4
        kernel_obj.base_kernel_parm5 = self.base_kernel_parm5
        kernel_obj.base_kernel_parm6 = self.base_kernel_parm6
        kernel_obj.base_kernel_parm7 = self.base_kernel_parm7
        kernel_obj.base_kernel_parm8 = self.base_kernel_parm8

        # Calculate sum of log-marginal likelihoods
        log_mll = 0.
        for idx in range(self.T):
            Y_l = self.Ys_tensor[idx,]             # 2d tensor, m * 1, the lth dataset
            X_l = self.Xs_tensor[idx,]             # 2d tensor, m * d, the lth dataset
            score_X_l = self.scores_Tensor[idx,]   # 2d tensor, m * d, the scores for all samples in the current dataset

            Y_l_b = Y_l[batch_sample_indices,:]    # b * 1
            X_l_b = X_l[batch_sample_indices,:]    # b * d
            score_X_l_b = score_X_l[batch_sample_indices,:]   # b * d

            k_Xlb_Xlb = kernel_obj.cal_stein_base_kernel(X_l_b, X_l_b, score_X_l_b, score_X_l_b) + beta_cstkernel

            cond_number_threshold = 1e6  # 1e3
            lam = 1e-6

            bad_cond = 1. / np.linalg.cond(k_Xlb_Xlb.detach().numpy()) < 10 ** (-6)
            k_Ylb = k_Xlb_Xlb + lam * torch.eye(X_l_b.size()[0])
            while bad_cond:
                lam = 10 * lam
                k_Ylb = k_Xlb_Xlb + lam * torch.eye(X_l_b.size()[0])
                bad_cond = np.linalg.cond(k_Ylb.detach().numpy()) >= cond_number_threshold
            print("regularization multiplier =", lam)
            k_Ylb.to(dtype=torch.float64)

            if Y_l_b.dim() == 1:
                Y_l_b = Y_l_b.unsqueeze(dim=1)

            distrib = torch.distributions.MultivariateNormal(torch.zeros(Y_l_b.size()[0]), covariance_matrix=k_Ylb)

            log_mll += distrib.log_prob(Y_l_b.squeeze())

        neg_log_mll = -1. * log_mll
        return neg_log_mll






class TuneKernelParams_mllk_MRI_multidats_borehole(object):
    def __init__(self, prior_kernel, base_kernel, Xs_tensor, Ys_tensor, scores_Tensor):
        """
        :param prior_kernel: a class
        :param base_kernel:  a class
        :param Xs_tensor:      3d Tensor, T * m * d ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param Ys_tensor:      3d Tensor, T * m * 1 ; when assuming m = m_1 = ... = m_T, i.e all datasets have same size
        :param scores_Tensor:  3d Tensor, T * m * d ; grad_x of logp(x) for m points from each of the dataset
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



    def do_optimize_logmll(self, batch_size, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.01, epochs=100, verbose=True):
        neg_mll = negative_log_marginal_lik_MRI_multidats_borehole(self.prior_kernel, self.base_kernel, batch_size, self.Xs_tensor, self.Ys_tensor, self.scores_Tensor, flag_if_use_medianheuristic)
        optimizer = torch.optim.Adam(neg_mll.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        m = self.Xs_tensor.size()[1]
        train_indices = list(range(m))
        for i in range(epochs):

            batches_generator = self.chunks(train_indices, batch_size)  # ZS: this creates a generator

            for batch_idx, batch_indices in enumerate(batches_generator):

                scheduler.step()
                optimizer.zero_grad()
                out = neg_mll(batch_indices, beta_cstkernel)
                out.backward()
                optimizer.step()

                with torch.no_grad():
                    for param in neg_mll.parameters():
                        param.clamp_(min=np.log(1e-3))

            if verbose:
                print(i + 1, iter, out, neg_mll.base_kernel_parm1.detach(), \
                          neg_mll.base_kernel_parm2.detach(), \
                          neg_mll.base_kernel_parm3.detach(), \
                          neg_mll.base_kernel_parm4.detach(), \
                          neg_mll.base_kernel_parm5.detach(), \
                          neg_mll.base_kernel_parm6.detach(), \
                          neg_mll.base_kernel_parm7.detach(), \
                          neg_mll.base_kernel_parm8.detach())

            # Random shuffle
            np.random.shuffle(train_indices)


        self.neg_mll = neg_mll












##########################################################################################
# 3. For multiple Unbalanced Datasets: SGD on sum of log marginal likelihood
##########################################################################################

class negative_log_marginal_lik_MRI_multidats_unbalanced_borehole(torch.nn.Module):
    def __init__(self, prior_kernel, base_kernel, batch_size, Xs_tuple, Ys_tuple, scores_tuple, flag_if_use_medianheuristic):
        """
        : Recall that we assume a zero mean function as prior mean function
        :param prior_kernel:   a class
        :param base_kernel:    a class
        :param Xs_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        :param Ys_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, 1] , ... , [mT, 1])
        :param scores_tuple:  a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        :param flag_if_use_medianheuristic: whether or not to use median heuristic
        """
        super(negative_log_marginal_lik_MRI_multidats_unbalanced_borehole, self).__init__()


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


        if flag_if_use_medianheuristic == False:
            self.base_kernel_parm1_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm2_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm3_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm4_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm5_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm6_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm7_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))
            self.base_kernel_parm8_raw = torch.nn.Parameter(torch.log(torch.ones(1, dtype=torch.float32, requires_grad=True) * 10))

        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel


        self.Xs_tuple = Xs_tuple
        self.Ys_tuple = Ys_tuple
        self.scores_tuple = scores_tuple

        self.batch_size = batch_size



    def forward(self, batch_sample_indices_dict, beta_cstkernel = 1):
        self.base_kernel_parm1 = torch.exp(self.base_kernel_parm1_raw)
        self.base_kernel_parm2 = torch.exp(self.base_kernel_parm2_raw)
        self.base_kernel_parm3 = torch.exp(self.base_kernel_parm3_raw)
        self.base_kernel_parm4 = torch.exp(self.base_kernel_parm4_raw)
        self.base_kernel_parm5 = torch.exp(self.base_kernel_parm5_raw)
        self.base_kernel_parm6 = torch.exp(self.base_kernel_parm6_raw)
        self.base_kernel_parm7 = torch.exp(self.base_kernel_parm7_raw)
        self.base_kernel_parm8 = torch.exp(self.base_kernel_parm8_raw)

        kernel_obj = self.prior_kernel(base_kernel=self.base_kernel)  # instantialized the class
        kernel_obj.base_kernel_parm1 = self.base_kernel_parm1
        kernel_obj.base_kernel_parm2 = self.base_kernel_parm2
        kernel_obj.base_kernel_parm3 = self.base_kernel_parm3
        kernel_obj.base_kernel_parm4 = self.base_kernel_parm4
        kernel_obj.base_kernel_parm5 = self.base_kernel_parm5
        kernel_obj.base_kernel_parm6 = self.base_kernel_parm6
        kernel_obj.base_kernel_parm7 = self.base_kernel_parm7
        kernel_obj.base_kernel_parm8 = self.base_kernel_parm8



        log_mll = 0.
        accum_ms = 0
        for idx in range(self.T):
            cur_indices = torch.tensor(batch_sample_indices_dict[idx]) + accum_ms
            Y_lb = self.Ys_flattened[cur_indices,:]  # 2d tensor, (c_l b) * 1, the lth dataset , where cl = m_list[l]/m_list.sum()
            X_lb = self.Xs_flattened[cur_indices,:]  # 2d tensor, (c_l b) * d, the lth dataset,  where cl = m_list[l]/m_list.sum()

            score_X_lb = self.scores_Tensor_flattened[cur_indices,:]  # 2d tensor, (c_l b) * d, where cl = m_list[l]/m_list.sum()

            k_Xlb_Xlb = kernel_obj.cal_stein_base_kernel(X_lb, X_lb, score_X_lb, score_X_lb) + beta_cstkernel

            cond_number_threshold = 1e6  # 1e3
            lam = 1e-6

            bad_cond = 1. / np.linalg.cond(k_Xlb_Xlb.detach().numpy()) < 10 ** (-6)
            k_Ylb = k_Xlb_Xlb + lam * torch.eye(X_lb.size()[0])
            while bad_cond:
                lam = 10 * lam
                k_Ylb = k_Xlb_Xlb + lam * torch.eye(X_lb.size()[0])
                bad_cond = np.linalg.cond(k_Ylb.detach().numpy()) >= cond_number_threshold
            print("regularization multiplier =", lam)
            k_Ylb.to(dtype=torch.float64)

            if Y_lb.dim() == 1:
                Y_lb = Y_lb.unsqueeze(dim=1)

            distrib = torch.distributions.MultivariateNormal(torch.zeros(Y_lb.size()[0]), covariance_matrix=k_Ylb)

            log_mll += distrib.log_prob(Y_lb.squeeze())

            accum_ms = accum_ms + self.m_list[idx]

        neg_log_mll = -1. * log_mll


        return neg_log_mll








class TuneKernelParams_mllk_MRI_multidats_unbalanced_borehole(object):
    def __init__(self, prior_kernel, base_kernel, Xs_tuple, Ys_tuple, scores_tuple):
        """
        :param prior_kernel: a class
        :param base_kernel:  a class
        :param Xs_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        :param Ys_tuple:      a tuple with T elements, whose each element is a 2d tensor, ([m1, 1] , ... , [mT, 1])
        :param scores_tuple:  a tuple with T elements, whose each element is a 2d tensor, ([m1, d] , ... , [mT, d])
        """

        self.prior_kernel = prior_kernel
        self.base_kernel = base_kernel

        self.Xs_tuple = Xs_tuple
        self.Ys_tuple = Ys_tuple
        self.scores_tuple = scores_tuple


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


    def do_optimize_logmll(self, batch_size, flag_if_use_medianheuristic, beta_cstkernel=1, lr=0.01, epochs=100, verbose=True):
        neg_mll = negative_log_marginal_lik_MRI_multidats_unbalanced_borehole(self.prior_kernel, self.base_kernel,batch_size, self.Xs_tuple, self.Ys_tuple, self.scores_tuple, flag_if_use_medianheuristic)
        optimizer = torch.optim.Adam(neg_mll.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        m_base = neg_mll.m_base
        train_indices = list(range(m_base))
        for i in range(epochs):
            batches_generator = self.chunks(train_indices, batch_size)

            for batch_idx, batch_indices in enumerate(batches_generator):
                # Use batch_indices to create a corresponding correct tuple of batch_indices
                batch_indices_dict = {}
                for cr in range(neg_mll.T):
                    start_pos = batch_idx * batch_size * (neg_mll.m_list[cr] / neg_mll.m_base)
                    end_pos = (batch_idx + 1) * batch_size * (neg_mll.m_list[cr] / neg_mll.m_base) - 1
                    batch_indices_dict[cr] = list(range(int(start_pos.item()), int(end_pos.item()) + 1))


                scheduler.step()
                optimizer.zero_grad()
                out = neg_mll(batch_indices_dict, beta_cstkernel)
                out.backward()
                optimizer.step()

                # set the params to be non-negative
                # Note the  torch.parameters here are on the log scale
                with torch.no_grad():
                    for param in neg_mll.parameters():
                        param.clamp_(min=np.log(1e-3))


            if verbose:
                print(i + 1, iter, out, neg_mll.base_kernel_parm1.detach(), \
                      neg_mll.base_kernel_parm2.detach(), \
                      neg_mll.base_kernel_parm3.detach(), \
                      neg_mll.base_kernel_parm4.detach(), \
                      neg_mll.base_kernel_parm5.detach(), \
                      neg_mll.base_kernel_parm6.detach(), \
                      neg_mll.base_kernel_parm7.detach(), \
                      neg_mll.base_kernel_parm8.detach())

            # Random shuffle
            np.random.shuffle(train_indices)

        self.neg_mll = neg_mll



