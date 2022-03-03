import pandas as pd
import seaborn as sns

from src.src_vvCV_MD1P.stein_operators import *
from src.src_vvCV_MDMP.stein_operators_mat_ker import *
from src.src_vvCV_MDMP.vv_CV_MDMP import *
from South_Function.South_function_trainer import *


# Example for vv_CV_MDMP
def my_func_1(X):
    return 1 + X+ X**2 + torch.sin(X * math.pi) * torch.exp(-1.* X.pow(2))


def my_func_2(X):
    return 1.5 + X+ 1.5*(X**2) + 1.75*torch.sin(X * math.pi) * torch.exp(-1.* X.pow(2))




mu_1_sit4 = torch.zeros(1,1)
cov_1_sit4= torch.eye(1)
mu_2_sit4 = torch.zeros(1,1)
cov_2_sit4= torch.eye(1) * 1.25
means_tuple_sit4 = (mu_1_sit4, mu_2_sit4)
covs_tuple_sit4 = (cov_1_sit4, cov_2_sit4)

tuple_of_meanscovstuple = ((means_tuple_sit4, covs_tuple_sit4))

true_vals = torch.Tensor([[2, 3.375]])
true_vals.size()
true_vals[0].size()


#
my_example = toy_example_MDMP(funcs= (my_func_1,  my_func_2), sample_size_per_dist = 50, num_rep = 100, \
                               vv_CV_model=VV_CV_vectorvaluedfuncs_model_MDMP, \
                               vv_CV_obj = penalized_ls_objective_vectorvaluedfunc_MDMP, \
                               prior_kernel = stein_matrix_valued_kernel , base_kernel=rbf_kernel, \
                               batch_size_tune = 5, flag_if_use_medianheuristic=False, beta_cstkernel=0, lr_tune=0.05,\
                               epochs_tune=30, verbose_tune=False, \
                               regularizer_const = 1e-3, regularizer_const_FB=1, batch_size=5, lr=1e-3, epochs=400, \
                               verbose=False)




# Run the algorithm and save outputs
MyvvCV_ests, MysvCV_ests, MysvCV_closed_form_sols = my_example.one_run(means_tuple_sit4, covs_tuple_sit4, (0,1))


#
T=2
m=50

#
vv_data_sorted_values, vv_data_sorted_indices = my_example.X_all.detach().squeeze().view(-1).sort()
sv_X1_data_sorted_values, sv_X1_data_sorted_indices = my_example.X_all[0,:,:].detach().squeeze().sort()  # X1 for f1
sv_X2_data_sorted_values, sv_X2_data_sorted_indices = my_example.X_all[1,:,:].detach().squeeze().sort()  # X2 for f2



test_x1 = torch.linspace(-4, 4, 100)
test_x1 = torch.unique(torch.sort(torch.cat((sv_X1_data_sorted_values, test_x1))).values)

test_x2 = torch.linspace(-4, 4, 100) + torch.randn(100)
test_x2 = torch.unique(torch.sort(torch.cat((sv_X2_data_sorted_values, test_x2))).values)


test_x1 = test_x1.unsqueeze(dim=1).unsqueeze(dim=0)
test_x1.size()

test_x2 = test_x2.unsqueeze(dim=1).unsqueeze(dim=0)
test_x2.size()

test_x = torch.cat((test_x1, test_x2), dim=0)
test_x


test_x1_sorted_values, test_x1_sorted_indices = test_x1.detach().squeeze().sort()
test_x2_sorted_values, test_x2_sorted_indices = test_x2.detach().squeeze().sort()


vv_theta = my_example.vvCV_model.fitting_obj.theta.detach().clone()
vv_est  = my_example.vvCV_model.fitting_obj.c.detach().clone().squeeze()
with torch.no_grad():
    # The giant 'Gram_Tensor' is a 6d Tensor, of size [T, T, m_1, m_2, T, T]
    vv_K0_XX = torch.zeros(T, T, test_x.size(1), m, T, T)

    score_tensor_all_test = torch.zeros(T, test_x.size(1), T, 1)
    for i in range(T):
        scores_cur_X_test = torch.zeros(test_x.size(1), T, 1)
        for j in range(T):
            scores_cur_X_test[:, j, :] = multivariate_Normal_score(means_tuple_sit4[j], covs_tuple_sit4[j], test_x[i])
        score_tensor_all_test[i, :, :, :] = scores_cur_X_test

    for i in range(T):
        for j in range(T):
            vv_K0_XX[i, j, :, :, :, :] = my_example.vvCV_model.fitting_obj.kernel_obj.cal_datasetwise(test_x[i], my_example.X_all[j], score_tensor_all_test[i], my_example.score_tensor_all[j])  # X, Z, score_tensor_X, score_tensor_Z


    vv_K0_XX = vv_K0_XX * my_example.vvCV_model.fitting_obj.B.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # To broadcast, self.B needs to match the dim of self.Gram_tensor, which is [T, T, m, m, T, T]

    vv_middle_1_forf1 = vv_K0_XX[0][:,:,:,0,:].reshape(T, test_x.size(1), -1)
    vv_middle_2_forf1 = vv_middle_1_forf1.transpose(0,1)
    vv_K0_XX_reshaped_forf1 = vv_middle_2_forf1.reshape(test_x.size(1),-1)
    vv_fitted_vals_f1 = vv_K0_XX_reshaped_forf1 @ vv_theta.view(T*m*T) + vv_est[0]

    vv_middle_1_forf2 = vv_K0_XX[1][:,:,:,1,:].reshape(T, test_x.size(1), -1)
    vv_middle_2_forf2 = vv_middle_1_forf2.transpose(0,1)
    vv_K0_XX_reshaped_forf2 = vv_middle_2_forf2.reshape(test_x.size(1),-1)
    vv_fitted_vals_f2 = vv_K0_XX_reshaped_forf2 @ vv_theta.view(T*m*T) + vv_est[1]




X1 = my_example.X_all[0]
X2 = my_example.X_all[1]
score_sv_X1 = multivariate_Normal_score(mu_1_sit4, cov_1_sit4, X1)
score_sv_X2 = multivariate_Normal_score(mu_2_sit4, cov_2_sit4, X2)
score_sv_testX1 = multivariate_Normal_score(mu_1_sit4, cov_1_sit4, test_x[0])
score_sv_testX2 = multivariate_Normal_score(mu_2_sit4, cov_2_sit4, test_x[1])

sv_theta_f1 = my_example.svCV_model_f1.fitting_obj.theta.detach().clone()
sv_est_f1 =  my_example.svCV_model_f1.fitting_obj.c.detach().clone()
with torch.no_grad():
    sv_kXX_f1 = my_example.svCV_model_f1.fitting_obj.kernel_obj.cal_stein_base_kernel(test_x[0], X1, score_sv_testX1,  score_sv_X1)
    sv_fitted_f1 = sv_kXX_f1 @  sv_theta_f1 + sv_est_f1


sv_theta_f2 = my_example.svCV_model_f2.fitting_obj.theta.detach().clone()
sv_est_f2 = my_example.svCV_model_f2.fitting_obj.c.detach().clone()
with torch.no_grad():
    sv_kXX_f2 = my_example.svCV_model_f2.fitting_obj.kernel_obj.cal_stein_base_kernel(test_x[1], X2, score_sv_testX2,  score_sv_X2)
    sv_fitted_f2 = sv_kXX_f2 @  sv_theta_f2 + sv_est_f2


#
a = np.linspace(-5, 5, 500)
b = 1 + a + a**2 + np.sin(np.pi * a ) * np.exp(-a**2)  # HF
c = 1.5 + a + 1.5 * (a**2) + 1.75 * np.sin(np.pi * a ) * np.exp(-a**2) # LF
my_vv_CV_DF_giant_f1f2_withoutCF = pd.read_pickle("../South_function_pdframe_saved.pkl")


sns.set_style("white")
fig, axs = plt.subplots(1,3, sharex=False, sharey=False)
fig.set_figwidth(20)
axs[0].set_title("Function 1", fontsize=20)
axs[1].set_title("Function 2", fontsize=20)

axs[0].plot(a, c, 'black', label = r'$f_1(x)$')
axs[1].plot(a, b, 'black', label = r'$f_2(x)$')

axs[0].set_ylabel("y", fontsize=20)
axs[1].set_ylabel("y", fontsize=20)
axs[0].set_ylim([0, 45])
axs[1].set_ylim([0, 45])

axs[0].set_xlabel("x", fontsize=20)
axs[1].set_xlabel("x", fontsize=20)


axs[0].tick_params(labelsize=20)
axs[1].tick_params(labelsize=20)

axs[0].plot(test_x2_sorted_values, vv_fitted_vals_f2[test_x2_sorted_indices], color='blue',  ls= 'dotted',  label = 'vv-CV')
axs[0].plot(test_x2_sorted_values, sv_fitted_f2[test_x2_sorted_indices], color='red' , ls= 'dotted', label = 'CV')


axs[1].plot(test_x1_sorted_values, vv_fitted_vals_f1[test_x1_sorted_indices], color='blue', ls= 'dotted', label = 'vv-CV')
axs[1].plot(test_x1_sorted_values, sv_fitted_f1[test_x1_sorted_indices], color='red' , ls= 'dotted', label = 'CV')




my_pal = {"vv-CV": "skyblue", "CV": "lightcoral"}
g=sns.boxplot(ax=axs[2],
              x="setting", #setting
              y="cv_est", # cv_est
              hue="method_idx", # method_idx
              data=my_vv_CV_DF_giant_f1f2_withoutCF,
              palette=my_pal,
              showmeans=True,
              meanprops={"marker": "+",
                         "linestyle": "--",
                         "color": "red",
                         "markeredgecolor": "red",
                         "markersize": "10"}
              )

g.set(ylim=(0.0001, None))
g.set(yscale="log") # xscale="log"
g.set(xlabel=None)
x_ticks_labels = [ r'$\sigma^2 =1$', r'$\sigma^2 =1.1$',r'$\sigma^2 =1.15$', r'$\sigma^2 =1.2$', r'$\sigma^2 =1.25$']
g.set(xticklabels=x_ticks_labels)
g.tick_params(labelsize=18)
g.set_ylabel("Sum of squared errors", fontsize=20)

axs[0].legend(fontsize=20)
axs[1].legend(fontsize=20)
axs[2].legend(fontsize = 20).set_title('')

plt.show()
# fig.savefig('filename.pdf')














