import scipy.io

from src.src_vvCV_MD1P.sv_CV import *
from src.src_vvCV_MDMP.vv_CV_MDMP import *



# Part I. Load useful data
data = scipy.io.loadmat('TI_Example/data.mat')
samples = scipy.io.loadmat('TI_Example/samples.mat')


# 1. Get Samples from the power posterior
theta = torch.from_numpy(samples['theta'])
theta = theta.squeeze()
theta.size()
theta.dtype
theta = theta.float()
theta.dtype

# 2. Get Linear control variates
z = torch.from_numpy(samples['z'])
z.size()
z.dtype
z = z.float()
z.dtype

# 3. Get Score function
# Note that the score function is u = -2 * z, where z is the linear control variate; for details see Eqn.6 of The controlled thermodynamic integral for Bayesian model comparison'
u = -2 * z
u = u.squeeze()
u.size()
u.dtype
u = u.float()
u.dtype



# 4. Get log-likelihood conditional on each sample
loglike = torch.from_numpy(samples['loglike'])
loglike = loglike.t()
loglike.size()
loglike.dtype
loglike = loglike.float()
loglike.dtype



# 5. Load other stuff
# -- From 'data'
theta_true = torch.from_numpy(data['theta_true']).squeeze()
theta_true = theta_true.float()
theta_true

time = torch.from_numpy(data['time']).squeeze()
time

sigma = torch.from_numpy(data['sigma']).squeeze()  # observational zero mean Gaussian noise with standard deviation = sigma
sigma
x0 = torch.from_numpy(data['x0']).squeeze()  # initial value of the agumented x = [x1 , x2]^\top, where x1 :=x, x2 := dx1/dt
x0
y = torch.from_numpy(data['y']).squeeze()    # observations
y


# -- From 'samples'
ti = torch.from_numpy(samples['ti']).squeeze()   # inverse temperatures
ti.dtype
ti = ti.float()
ti.dtype

N = torch.from_numpy(samples['N'].astype(np.float16)).squeeze() # samples from each power posterior --> so the total number of samples is N * number_of_ti
N
burn = torch.from_numpy(samples['burn'].astype(np.float16)).squeeze() # burn for samples from each power posterior
burn



# Plot scores to detach relationships
fig_u, axs_u = plt.subplots(8,4, sharex=True, sharey=True)
fig_u.set_figwidth(12)
fig_u.set_figheight(12)


show_u_indx = np.arange(0,505,5)
show_u_indx = show_u_indx-1
show_u_indx[0] = 0
show_u_indx

for i in range(8):
    for j in range(4):
        plot_u_idx = 8 * j + i
        if plot_u_idx > 30:
            break

        axs_u[i, j].plot(theta[plot_u_idx,:], u[plot_u_idx,:].detach().numpy(),'bo')
        axs_u[i, j].set_xlabel(r'$\theta$', fontsize=11)
        if j == 0:
            axs_u[i, j].set_ylabel(r'$\nabla_{\theta} \log p(\theta|y,t)$', fontsize=15)
        axs_u[i, j].set_title('Temp t = {:.3e}'.format(ti.detach().numpy()[plot_u_idx]), fontsize=15)


axs_u[7,3].set_axis_off()
plt.show()
plt.close()
# fig_u.savefig('figurename.pdf')





##
# Part 2. MRI for \{mu_i, v_i\}, for all i and Compute model evidence using TI and vv-CV

# Some important message:
# : Use Var[X] = E[X^2] - (E[X])^2 to calculate v_i
# : Note that in oates code of his JRSSB paper, everytime, he sampled 20 samples from each temperature. He repeated this 100 times to get the histogram.



class TI_vvCV(object):
    def __init__(self, ti, theta, loglike, scores, vv_CV_model, vv_CV_obj, prior_kernel, base_kernel, beta_cstkernel, batch_size_tune, flag_if_use_medianheuristic, lr_tune, epochs_tune, verbose_tune, regularizer_const, regularizer_const_FB, batch_size, lr, epochs, verbose):
        """
        :param ti:       1d Tensor of size 31; record all temperatures, each temp has a related power posterior
        :param theta:    2d Tensor of size [31, 500]; record all samples from each power posterior
        :param loglike:  2d Tensor of size [31, 500]; record the log-likelihood for each sample from every temperature., i.e. \logp(y|\theta_j), j \in [31*500]
        :param scores:   2d Tensor of size [31, 500]; record the score for each sample from every temperature, e.g. \nabla_theta_{i,j} \logp(\theta_{i,j}|y, t_j) for j \in [31], for i \in [500]

        :param vv_CV_model: a class; training model class
        :param vv_CV_obj:   a class; objective class
        :param prior_kernel: a class;  here should be 'stein_matrix_valued_kernel'
        :param base_kernel:  a class of scalar-valued kernel
        :param beta_cstkernel: a constant add to kernel evaluations; e.g., 1
        :param batch_size_tune: batch size for tuning the kernel hyper-parameters, e.g., 5
        :param flag_if_use_medianheuristic: bool; if true means we will use median heuristic for lengthscale
        :param lr_tune: learning rate for kernel hyperparameter tuning; e.g., 0.01
        :param epochs_tune: total number of epochs for tuning kernel hype-parameters e.g, 20
        :param verbose_tune: bool; e.g., 'True' to print relevant info
        :param regularizer_const: a scalar value for theta; e.g, 1e-3
        :param regularizer_const_FB: THIS NEEDS TO BE FIXED TO BE 1.
        :param batch_size: a scalar value, e.g., 5; batch size for optimizing the vvCV objective, when we have T tasks, for each task, we take 'batch_size' samples from the corresponding dataset.
        :param lr: a scalar value, e.g., 0.001
        :param epochs: a scalar value, e.g., 100
        :param verbose: bool; e.g., True
        """
        assert regularizer_const_FB == 1, 'regularizer_const_FB should be fixed to 1.'

        self.ti = ti
        self.theta = theta
        self.loglike = loglike
        self.scores = scores

        self.vv_CV_model, self.vv_CV_obj = vv_CV_model, vv_CV_obj
        self.prior_kernel, self.base_kernel, self.beta_cstkernel = prior_kernel, base_kernel, beta_cstkernel
        self.batch_size_tune, self.flag_if_use_medianheuristic, self.lr_tune, self.epochs_tune, self.verbose_tune = batch_size_tune, flag_if_use_medianheuristic, lr_tune, epochs_tune, verbose_tune
        self.regularizer_const, self.regularizer_const_FB, self.batch_size, self.lr, self.epochs, self.verbose = regularizer_const, regularizer_const_FB, batch_size, lr, epochs, verbose



    def score_log_Gaussian_prior(self, X, mu, sigma):
        """
        x \sim logNormal(mu, sigma^2)  <==> logX \sim Normal(mu, sigma^2)
        :param x:  2d Tensor, a point of size [m, d=1]
        :param mu: 2d Tensor of size  [1,1].     Here we should set mu = 0
        :param sigma: 2d Tensor of size [1,1]    Here we should set sigma = 0.25
        :return: 2d Tensor [m, d=1]
        """
        m = X.size(0)
        score_at_X = - 1. / X - (torch.log(X) - mu) / (X * (sigma ** 2))
        assert score_at_X.size() == torch.Size([m, 1])
        return score_at_X


    def cal_scores_at_difftemp_given_samplesoftheta(self, cur_temp, another_temp, thetas_from_cur_temp, scores_from_cur_temp):
        """
        :param cur_temp:              a value
        :param another_temp:          a value
        :param thetas_from_cur_temp:  2d Tensor of size [m, d]; some samples of theta from current temperature
        :param scores_from_cur_temp:  2d Tensor of size [m, d]; the corresponding scores of those samples \nabla_\theta \log[p(\theta|y, t)]
        :return:    2d Tensor of size [m, d]
        """
        m = thetas_from_cur_temp.size(0)
        d = thetas_from_cur_temp.size(1)  # this should be 1 actually

        prior_scores = self.score_log_Gaussian_prior(thetas_from_cur_temp, 0, 0.25)
        assert prior_scores.size() == torch.Size([m, d])
        assert scores_from_cur_temp.size() == torch.Size([m, d])

        factor = torch.Tensor([another_temp/cur_temp])
        factor = factor.unsqueeze(dim=0) # 2d Tensor of size [1,1] --- Zhuo: this step is for proper broadcasting

        scores_at_a_diff_temp = factor * (scores_from_cur_temp - prior_scores) + prior_scores
        assert scores_at_a_diff_temp.size() == torch.Size([m, d])

        return scores_at_a_diff_temp



    def thermodynamic_integral_svCV(self, num_samples_per_temp):
        """
        :param num_samples_per_temp:
        :return:
        """
        num_temps = self.ti.size(0)

        perm = torch.randperm(self.theta.size(1))  # which is 500
        idx = perm[:num_samples_per_temp]
        theta_samples   = self.theta[:, idx]       # 2d Tensor; [31, num_samples_per_temp], e.g., [31, 20]
        loglike_samples = self.loglike[:, idx]     # 2d Tensor; [31, num_samples_per_temp], e.g., [31, 20]
        scores_samples  = self.scores[:, idx]      # 2d Tensor; [31, num_samples_per_temp], e.g., [31, 20]

        # Step 1, get \{\mu_i, v_i\}, for i = 1, ..., num_temps (31)
        Eloglike = torch.zeros(num_temps)
        E2loglike = torch.zeros(num_temps)
        Vloglike = torch.zeros(num_temps)

        for i in range(num_temps):
            print("sv-CVs --- current {}th/31 temperature".format(i+1))
            cur_score_X = scores_samples[i,:].unsqueeze(dim=1) # ensure a 2d Tensor of size [20, 1], since theta is 1-dimensional
            cur_X = theta_samples[i,:].unsqueeze(dim=1)        # ensure a 2d Tensor of size [20, 1], since theta is 1-dimensional
            cur_Y = loglike_samples[i,:].unsqueeze(dim=1)      # ensure a 2d Tensor of size [20, 1]
            cur_Y_squared = cur_Y**2

            # Get E(X)
            my_model_get_mu = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, self.base_kernel, cur_X, cur_Y, cur_score_X)


            my_model_get_mu.optim_base_kernel_parms = torch.Tensor([0.1, 3])


            my_model_get_mu.do_optimize_sv_CV(regularizer_const=self.regularizer_const, batch_size=self.batch_size, \
                                              lr=self.lr, epochs=self.epochs, verbose=self.verbose)

            Eloglike[i] = my_model_get_mu.saved_BQ_est[-1].detach().clone()


            # Get E(X^2)
            my_model_get_mu_squared = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, self.base_kernel, cur_X, cur_Y_squared, cur_score_X)

            my_model_get_mu_squared.optim_base_kernel_parms = torch.Tensor([0.1, 3])


            my_model_get_mu_squared.do_optimize_sv_CV(regularizer_const=self.regularizer_const, batch_size=self.batch_size, \
                                              lr=self.lr, epochs=self.epochs, verbose=self.verbose)



            E2loglike[i] = my_model_get_mu_squared.saved_BQ_est[-1].detach().clone()


            # Get V(X)
            Vloglike[i] = E2loglike[i] - (Eloglike[i])**2


        # Step 2. Compute TI_sv_CV_estimator
        TI_sv_CV_estimator = torch.zeros(1)
        for t in range(num_temps):
            if t == 30:
                continue
            adjacent_ti_diff = self.ti[t + 1] - self.ti[t]
            adjacent_avg_mus = (Eloglike[t + 1] + Eloglike[t]) / 2.
            adjacent_var_diff = (Vloglike[t + 1] - Vloglike[t]) / 12.

            TI_sv_CV_estimator = TI_sv_CV_estimator + adjacent_ti_diff * adjacent_avg_mus - adjacent_var_diff * (adjacent_ti_diff ** 2)

        self.Eloglike_sv = Eloglike
        self.E2loglike_sv = E2loglike
        self.Vloglike_sv = Vloglike
        self.TI_sv_CV_estimator = TI_sv_CV_estimator

        return TI_sv_CV_estimator






    def thermodynamic_integral_vvCV(self, num_samples_per_temp):
        """
        :param ti:       1d Tensor of size 31
        :param theta:    2d Tensor of size [31, 500]
        :param loglike:  2d Tensor of size [31, 500]
        :param scores:   2d Tensor of size [31, 500]
        :return:
        """
        num_temps = self.ti.size(0)

        perm = torch.randperm(self.theta.size(1))
        idx = perm[:num_samples_per_temp]
        theta_samples   = self.theta[:, idx]     # 2d Tensor; [31, num_samples_per_temp], e.g., [31, 20]
        loglike_samples = self.loglike[:, idx]   # 2d Tensor; [31, num_samples_per_temp], e.g., [31, 20]
        scores_samples  = self.scores[:, idx]    # 2d Tensor; [31, num_samples_per_temp], e.g., [31, 20]


        ##
        # Step 1, get \{\mu_i, v_i\}, for i = 1, ..., num_temps (31)
        Eloglike = torch.zeros(num_temps)
        E2loglike = torch.zeros(num_temps)
        Vloglike = torch.zeros(num_temps)

        count_tasks_so_far = 0 # To monitor if we get all mu_i and v_i's

        for i in range(np.int(np.floor(31 / 4)+1)  ):
            if i == 7:
                num_tasks_vvCV = 3  #
            else:
                num_tasks_vvCV = 4  #



            ## Record current task indices
            cur_temp_indices = count_tasks_so_far + torch.arange(num_tasks_vvCV)
            # print(cur_temp_indices)
            print("vv-CVs --- current {}th/31 temperature".format(cur_temp_indices + 1))

            # Get current task temperatures
            cur_temps = self.ti[cur_temp_indices]     # 1d Tensor of size [len(cur_temp_indices)], e.g., when i=0, [3]; when i neq 0, [2]


            # Prepare datapoints and responses
            xall = theta_samples[cur_temp_indices, :]
            xall = xall.unsqueeze(dim=2)                # 3d Tensor of size [len(cur_temp_indices), num_samples_per_temp], e.g., when i=0, [3, 20, 1]; when i neq 0, [2, 20, 1]
            yall = loglike_samples[cur_temp_indices, :]
            yall = yall.unsqueeze(dim=2)                # 3d Tensor of size [len(cur_temp_indices), num_samples_per_temp], e.g., when i=0, [3, 20, 1]; when i neq 0, [2, 20, 1]
            yall_squared = yall**2                      # 3d Tensor of size [len(cur_temp_indices), num_samples_per_temp], e.g., when i=0, [3, 20, 1]; when i neq 0, [2, 20, 1]

            # Prepare scores
            scores_tensor =  scores_samples[cur_temp_indices, :]   # 2d Tensor; [len(cur_temp_indices), num_samples_per_temp], e.g., when i=0, [3, 20]; when i neq 0, [2, 20]


            score_cross_tensor = torch.zeros(len(cur_temps), num_samples_per_temp, len(cur_temps), 1)   # This should be T * m * T * d; T tasks, each has m points, each point has a score matrix of T-by-d
            for j in range(len(cur_temps)):
                for k in range(len(cur_temps)):
                    if j == k:
                        score_cross_tensor[j, :, k, :] = scores_tensor[j, :].unsqueeze(dim=1)

                    if k != j:
                        cur_temp = cur_temps[j]
                        another_temp = cur_temps[k]
                        thetas_from_cur_temp = theta_samples[j, :]
                        thetas_from_cur_temp = thetas_from_cur_temp.unsqueeze(dim=1)  # ensure a 2d Tensor [20, 1]
                        scores_from_cur_temp = scores_samples[j, :]
                        scores_from_cur_temp = scores_from_cur_temp.unsqueeze(dim=1)  # ensure a 2d Tensor [20,1]

                        # Worth a double check
                        score_cross_tensor[j, :, k, :] = self.cal_scores_at_difftemp_given_samplesoftheta(cur_temp, another_temp, thetas_from_cur_temp, scores_from_cur_temp)  # LHS is a 2d Tensor of size [m, d]



            # Get E(X)
            my_model_get_mu = self.vv_CV_model(vv_cv_objective = self.vv_CV_obj, prior_kernel = self.prior_kernel, base_kernel = self.base_kernel, Xs_tensor = xall, Ys_tensor = yall, scores_Tensor = score_cross_tensor)
            # my_model_get_mu.do_tune_kernelparams_negmllk(batch_size_tune= self.batch_size_tune, flag_if_use_medianheuristic=self.flag_if_use_medianheuristic, beta_cstkernel=self.beta_cstkernel, lr=self.lr_tune, epochs=self.epochs_tune, verbose=self.verbose_tune)

            my_model_get_mu.optim_base_kernel_parms = torch.Tensor([0.1, 3])
            my_model_get_mu.do_optimize_vv_CV(regularizer_const = self.regularizer_const, regularizer_const_FB = self.regularizer_const_FB, batch_size = self.batch_size, lr=self.lr, epochs=self.epochs, verbose=self.verbose)

            Eloglike[cur_temp_indices] =  my_model_get_mu.saved_BQ_est[-1,:,:].squeeze().detach().clone()


            # Get E(X^2)
            my_model_get_mu_squared = self.vv_CV_model(vv_cv_objective=self.vv_CV_obj, prior_kernel=self.prior_kernel,base_kernel=self.base_kernel, Xs_tensor=xall, Ys_tensor=yall_squared, scores_Tensor=score_cross_tensor)

            my_model_get_mu_squared.optim_base_kernel_parms = torch.Tensor([0.1, 3])
            my_model_get_mu_squared.do_optimize_vv_CV(regularizer_const=self.regularizer_const,\
                                              regularizer_const_FB=self.regularizer_const_FB,\
                                              batch_size=self.batch_size, lr=self.lr, epochs=self.epochs,\
                                              verbose=self.verbose)

            E2loglike[cur_temp_indices] = my_model_get_mu_squared.saved_BQ_est[-1, :, :].squeeze().detach().clone()


            # Get V(X)
            Vloglike[cur_temp_indices] = E2loglike[cur_temp_indices] - (Eloglike[cur_temp_indices])**2


            # Accum
            count_tasks_so_far += num_tasks_vvCV


        # Assure total number of tasks is 31
        assert count_tasks_so_far == 31, 'Total number of evaluations mismatch 31.'



        # Step 2. Compute TI_vv_CV_estimator
        TI_vv_CV_estimator = torch.zeros(1)
        for t in range(num_temps):
            if t == 30:
                continue
            adjacent_ti_diff = self.ti[t+1] - self.ti[t]
            adjacent_avg_mus = (Eloglike[t+1] + Eloglike[t])/2.
            adjacent_var_diff = (Vloglike[t+1] - Vloglike[t])/12.

            TI_vv_CV_estimator = TI_vv_CV_estimator + adjacent_ti_diff * adjacent_avg_mus - adjacent_var_diff * (adjacent_ti_diff**2)



        self.Eloglike_vv = Eloglike
        self.E2loglike_vv = E2loglike
        self.Vloglike_vv = Vloglike
        self.TI_vv_CV_estimator = TI_vv_CV_estimator


        return TI_vv_CV_estimator









# # Experiments
ti_modified = ti
ti_modified[0] = ti_modified[0] + 1e-16
ti_modified


n_rep = 20
ss_set = [20, 40, 60, 80]
I_vvCV = torch.zeros(n_rep, len(ss_set))
I_svCV = torch.zeros(n_rep, len(ss_set))



for j in range(len(ss_set)): # for j in range([2,3,4]):

    for i in range(n_rep):
        torch.manual_seed(n_rep * 0 + i)
        print("seed is {}".format(n_rep * j + i))

        my_TI_vvCV = TI_vvCV(ti=ti_modified, theta=theta, loglike=loglike, scores=u,\
                             vv_CV_model=VV_CV_vectorvaluedfuncs_model_MDMP, \
                             vv_CV_obj=penalized_ls_objective_vectorvaluedfunc_MDMP, \
                             prior_kernel=stein_matrix_valued_kernel, base_kernel=base_kernel_2, beta_cstkernel=1, \
                             batch_size_tune=5, flag_if_use_medianheuristic=False, \
                             lr_tune=0.03, epochs_tune=20, verbose_tune=False, \
                             regularizer_const=1e-3, regularizer_const_FB=1, batch_size=5, lr=0.01, epochs=400, \
                             verbose=False)

        I_svCV[i, j] = my_TI_vvCV.thermodynamic_integral_svCV(num_samples_per_temp=ss_set[j])

        I_vvCV[i,j] = my_TI_vvCV.thermodynamic_integral_vvCV(num_samples_per_temp=ss_set[j])

        print(i, j, I_vvCV[i,j],I_svCV[i,j] )


    fig, axs = plt.subplots(1,2, sharex=True, sharey=True)
    axs[0].set_title("vv-CVs")
    axs[1].set_title("sv-CVs")
    axs[0].set_ylabel('Model Evidence', fontsize=12)
    fig.set_figwidth(10)
    axs[0].boxplot(I_vvCV[:,j].detach().numpy())
    axs[1].boxplot(I_svCV[:,j].detach().numpy())


    x_ticks_labels = ['SS {}'.format(ss_set[j])]
    axs[0].set_xticks([1])
    axs[1].set_xticks([1])
    # Set ticks labels for x-axis
    axs[0].set_xticklabels(x_ticks_labels, fontsize=10) #rotation='vertical'
    axs[1].set_xticklabels(x_ticks_labels, fontsize=10) #rotation='vertical'

    plt.show()
    plt.close()





##################
# Save output
##################
# If you want to rerun the above experiment and save your own results, please uncomment the following line to save your data.

# with open('..data/TI_example_all_data.pkl', 'wb') as output:
#     I_svCV = I_svCV
#     pickle.dump(I_svCV, output, pickle.HIGHEST_PROTOCOL)
#
#     I_vvCV = I_vvCV
#     pickle.dump(I_vvCV, output, pickle.HIGHEST_PROTOCOL)







