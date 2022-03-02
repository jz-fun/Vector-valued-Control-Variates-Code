from src.src_vvCV_MD1P.sv_CV import *
from src.src_vvCV_MD1P.stein_operators import *
from src.src_vvCV_MD1P.base_kernels import *
from src.src_vvCV_MDMP.vv_CV_MDMP import *
from src.src_vvCV_MDMP.utils_for_stein_mat_ker import *


class toy_example_MDMP(object):
    def __init__(self, funcs, sample_size_per_dist, num_rep, vv_CV_model, vv_CV_obj, prior_kernel, base_kernel, \
                 batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr_tune, epochs_tune, verbose_tune, \
                 regularizer_const, regularizer_const_FB, batch_size, lr, epochs, verbose):

        assert regularizer_const_FB==1, "regularizer_const_FB should be 1 to avoid over parametrization."

        self.funcs, self.sample_size_per_dist, self.num_rep = funcs, sample_size_per_dist, num_rep
        self.vv_CV_model, self.vv_CV_obj, self.prior_kernel, self.base_kernel = vv_CV_model, vv_CV_obj, prior_kernel, base_kernel
        self.batch_size_tune, self.flag_if_use_medianheuristic, self.beta_cstkernel, self.lr_tune, self.epochs_tune, self.verbose_tune = batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr_tune, epochs_tune, verbose_tune
        self.regularizer_const, self.regularizer_const_FB, self.batch_size, self.lr, self.epochs, self.verbose = regularizer_const, regularizer_const_FB, batch_size, lr, epochs, verbose

        self.T = len(funcs)


    def sample_and_get_scores(self, number_samples, mean_Gaussians, cov_Gaussians, randn_seeds_tuple):
        """
        :param number_samples: a value; e.g, 20
        :param mean_Gaussian: a tuple of 2d Tensor of size [d, 1]; mean (vector) of a (multivariate) Gaussian distribution
        :param cov_Gaussian:  a tuple of 2d Tensor of size [d, d]; cov-variance (matrix) of a (multivariate) Gaussian distribution
        :param randn_seeds_tuple: a tuple of random seeds of size T, e.g., (1,2,3,4,... T).
                                  This is for reproducing the experiments.
                                  (which is used for generating random datasets inside the algorithm)
        :return: samples and their cross-over scores
        """

        T = len(mean_Gaussians)
        _ = len(cov_Gaussians)
        assert T == _ , "Contain different number of mean vectors and cov-matrices. Check your input."

        d = mean_Gaussians[0].size(0)
        m = number_samples


        #
        score_tensor_all = torch.zeros(T, m, T, d)
        X_all = torch.zeros(T, m, d)
        Y_all = torch.zeros(T, m, 1)

        for i in range(T):
            cur_func = self.funcs[i]
            cur_mean_vec = mean_Gaussians[i] # 2d Tensor of size [d,1]
            cur_cov_mat = cov_Gaussians[i]   # 2d Tensor of size [d,d]
            assert torch.det(cur_cov_mat) > 0, "The {}th cov_Gaussian should be PSD. Check your input.".format(i+1)

            torch.manual_seed(randn_seeds_tuple[i])
            cur_Z= torch.randn(m, d)
            cur_L = torch.cholesky(cur_cov_mat) # 2d Tensor of size [d, d]

            cur_X = (cur_Z @ cur_L + cur_mean_vec.t())
            assert cur_X.size() == torch.Size([m ,d])

            cur_Y = cur_func(cur_X)
            assert cur_Y.size() == torch.Size([m, 1])

            scores_cur_X = torch.zeros(m, T, d)
            for j in range(T):
                scores_cur_X[:, j, :] = multivariate_Normal_score(mean_Gaussians[j], cov_Gaussians[j], cur_X)


            X_all[i,:,:] = cur_X
            Y_all[i,:,:] = cur_Y
            score_tensor_all[i,:,:,:] = scores_cur_X



        return X_all, Y_all, score_tensor_all, T




    def one_run(self, mean_Gaussians, cov_Gaussians, randn_seeds_tuple):
        """
        :param mean_Gaussian: a tuple of 2d Tensor of size [d, 1]; mean (vector) of a (multivariate) Gaussian distribution
        :param cov_Gaussian:  a tuple of 2d Tensor of size [d, d]; cov-variance (matrix) of a (multivariate) Gaussian distribution
        :param randn_seeds_tuple: a tuple of random seeds of size T, e.g., (1,2,3,4,... T)
        :return:
        """
        X_all, Y_all, score_tensor_all, T = self.sample_and_get_scores(self.sample_size_per_dist, mean_Gaussians, cov_Gaussians, randn_seeds_tuple)


        vvCV_model = self.vv_CV_model(vv_cv_objective=self.vv_CV_obj, prior_kernel=self.prior_kernel,\
                                                      base_kernel=self.base_kernel, Xs_tensor=X_all, Ys_tensor=Y_all,\
                                                      scores_Tensor=score_tensor_all)
        vvCV_model.do_tune_kernelparams_negmllk(batch_size_tune=self.batch_size_tune, \
                                              flag_if_use_medianheuristic=self.flag_if_use_medianheuristic, \
                                              beta_cstkernel=self.beta_cstkernel,\
                                              lr=self.lr_tune, epochs=self.epochs_tune, verbose=self.verbose_tune)
        vvCV_model.do_optimize_vv_CV(regularizer_const=self.regularizer_const, \
                                   regularizer_const_FB=self.regularizer_const_FB, batch_size=self.batch_size, \
                                   lr= self.lr, epochs=self.epochs, verbose=self.verbose)


        vvCV_ests = vvCV_model.saved_BQ_est.squeeze()[-1].detach().clone()



        svCV_ests = torch.zeros(T)
        svCV_closed_form_sols = torch.zeros(T)

        for i in range(T):
            print("Current {}th task (sv-CV).".format(i+1))
            cur_X = X_all[i] # 2d Tensor of size [m, d]
            cur_Y = Y_all[i] # 2d Tensor of size [m, d]
            cur_score = score_tensor_all[i,:,i,:] # 2d Tensor of size [m, d]


            svCV_model = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc,\
                                                       stein_base_kernel_MV_2, self.base_kernel, \
                                                       cur_X, cur_Y, cur_score)


            svCV_model.do_tune_kernelparams_negmllk(batch_size_tune=self.batch_size_tune , \
                                                    flag_if_use_medianheuristic=self.flag_if_use_medianheuristic,\
                                                    beta_cstkernel=self.beta_cstkernel, lr=self.lr_tune, \
                                                    epochs=self.epochs_tune, verbose=self.verbose_tune)

            sv_bs_opt = self.batch_size * T
            svCV_model.do_optimize_sv_CV(regularizer_const=self.regularizer_const, batch_size=sv_bs_opt, lr=self.lr,\
                                         epochs=self.epochs, verbose=self.verbose)

            svCV_ests[i] = svCV_model.saved_BQ_est[-1].detach().clone()
            svCV_closed_form_sols[i] = svCV_model.do_closed_form_est_for_simpliedCF()


            # Save for plot -- not necessary
            if i ==0:
                self.svCV_model_f1 = svCV_model
            if i ==1:
                self.svCV_model_f2 = svCV_model


        self.vvCV_model = vvCV_model
        self.X_all = X_all
        self.Y_all = Y_all
        self.score_tensor_all = score_tensor_all

        return vvCV_ests, svCV_ests, svCV_closed_form_sols




    def multi_runs(self, mean_Gaussians, cov_Gaussians):
        """
        :param mean_Gaussians:
        :param cov_Gaussians:
        :return:
        """
        vvCV_ests_all = torch.zeros(self.num_rep, self.T)
        svCV_ests_all = torch.zeros(self.num_rep, self.T)
        svCV_closed_form_all = torch.zeros(self.num_rep, self.T)

        for i in range(self.num_rep):
            print("Now the {}th rep".format(i+1))
            randn_seeds_tuple = tuple([self.T * i + k for k in list(range(self.T))])

            print(randn_seeds_tuple)

            vvCV_ests_all[i], svCV_ests_all[i], svCV_closed_form_all[i] = self.one_run(mean_Gaussians, cov_Gaussians, randn_seeds_tuple)

        return vvCV_ests_all, svCV_ests_all, svCV_closed_form_all



    def varying_distrbutions_multiruns(self, Tuple_of_means_covs_tuple):
        """
        :param Tuple_of_means_covs_tuple:  a tuple contains (means_tuple_sit_1, cov_tuple_sit_1), ...., (means_tuple_sit_i, cov_tuple_sit_i),
                                           where (means_tuple_sit_i, cov_tuple_sit_i) is a tuple contains two sub-tuples under ith situation, the first one is a tuple with means for each distributions;
                                           the second one is a tuple with covmatrices for each distributions.
        :return:
        """
        num_situations = len(Tuple_of_means_covs_tuple)

        _ = len(Tuple_of_means_covs_tuple[0][0])
        assert self.T == _, "The number of functions mistaches the number of mean vectors. Check your input."

        # Each situation, we have a tuple of means and a tuple of cov-matrices for T tasks, each situation we repeat self.num_rep times.
        vvCV_ests_allsits = torch.zeros(num_situations, self.num_rep, self.T)                 # 3d Tensor of size [num_sits, num_rep, T]
        svCV_ests_allsits = torch.zeros(num_situations, self.num_rep, self.T)                 # 3d Tensor of size [num_sits, num_rep, T]
        svCV_closed_form_sols_allsits = torch.zeros(num_situations, self.num_rep, self.T)     # 3d Tensor of size [num_sits, num_rep, T]

        for i in range(num_situations):
            print("Current is {}th situation".format(i+1))
            cur_sit_means_tuple = Tuple_of_means_covs_tuple[i][0]
            cur_sit_covs_tuple  = Tuple_of_means_covs_tuple[i][1]
            vvCV_ests_allsits[i], svCV_ests_allsits[i], svCV_closed_form_sols_allsits[i] = self.multi_runs(cur_sit_means_tuple, cur_sit_covs_tuple)

        return vvCV_ests_allsits, svCV_ests_allsits, svCV_closed_form_sols_allsits










