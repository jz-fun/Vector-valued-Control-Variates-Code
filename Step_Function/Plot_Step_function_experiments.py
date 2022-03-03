
import pickle
from matplotlib import pyplot as plt
import torch
import seaborn as sns
import numpy as np

from src.src_vvCV_MD1P.stein_operators import *
from src.src_vvCV_MD1P.sv_CV import *
from src.src_vvCV_MD1P.vv_CV_MD1P import *
from src.src_vvCV_MD1P.vv_CV_FixB_MD1P import *
from src.src_vvCV_MD1P.vv_CV_unbalanced_FixB_MD1P import *



# ======================
# Step Function
# ======================

# Set  vv-CV kernel
my_base_kernel = rbf_kernel
my_lr = 0.0003
my_poly_ker_parm = torch.Tensor([1,1])
no_replica_ty2 = 1
no_epochs_ty2 = 400
no_points_per_func_ty2= 40



for i in range(no_replica_ty2):
    print("REP {} out of {}-----------".format(i+1, no_replica_ty2 ))
    dim = 1
    factor = torch.ones(1) * 1
    mu = torch.zeros(dim, dtype=torch.float) + 0
    var = torch.eye(dim, dtype=torch.float) * factor  # MUst use eye() here
    print(mu, var)

    def my_func_1(X):
        return (0.5 + (2 * (X >= 0) - 1) * 1.5) * torch.ones(1, dtype=torch.float)



    def my_func_2(X):
        return (X >= 0) * torch.ones(1, dtype=torch.float)


    # Training samples
    print("REP {} out of {}-----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(5)
    X1 = mu + torch.sqrt(factor) * torch.randn(no_points_per_func_ty2, dim)
    Y1 = my_func_1(X1)


    # --- For MD1P
    torch.manual_seed(6)
    X2 = mu + torch.sqrt(factor) * torch.randn(no_points_per_func_ty2, dim)
    Y2 = my_func_2(X2)



    # --- For 1D1P
    Y1_X2 = my_func_1(X2)
    Ys_on_X2 = torch.stack((Y1_X2, Y2), dim=1).squeeze()


    # Scores on X's
    mu = torch.zeros(dim, 1)
    cov = var
    score_X1 = multivariate_Normal_score(mu, cov, X1)
    score_X1.size()
    score_X2 = multivariate_Normal_score(mu, cov, X2)

    xall = torch.stack((X1, X2), dim=0)
    xall.size()
    yall = torch.stack((Y1, Y2), dim=0)
    yall.size()
    score_all = torch.stack((score_X1, score_X2), dim=0)
    score_all.size()



    # f1
    print("REP {} out of {} --- sv-CV-f1 -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc1 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, my_base_kernel, X1, Y1, score_X1)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc1.do_tune_kernelparams_negmllk(batch_size_tune = 5, flag_if_use_medianheuristic=False,  beta_cstkernel=0., lr=0.02, epochs=15, verbose=True)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc1.do_optimize_sv_CV(regularizer_const = 1e-5, batch_size = 10, lr = my_lr, epochs = no_epochs_ty2, verbose = True)


    # f2
    print("REP {} out of {}--- sv-CV-f2 -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc2 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc,stein_base_kernel_MV_2, my_base_kernel, X2, Y2, score_X2)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc2.do_tune_kernelparams_negmllk(batch_size_tune = 5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc2.do_optimize_sv_CV(regularizer_const=1e-5, batch_size=10, lr=my_lr, epochs=no_epochs_ty2, verbose=True)



    # vv-CV: MD1P with B fixed
    print("REP {} out of {} --- vv-CV: MD1P with B fixed -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=my_base_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB.do_tune_kernelparams_negmllk(batch_size_tune=5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True) # bs 5; lr 0.2; epochs 5
    torch.manual_seed(0)
    # set B
    Ty2_SCV_vectorvaluedfunc_fixB.B =  torch.Tensor([[0.1, 0.01], [0.01,0.1]])
    Ty2_SCV_vectorvaluedfunc_fixB.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)



    # ---------------
    # vv-CV: MD1P with B fixed --  ANOTHER B
    print("REP {} out of {} --- vv-CV: MD1P with B fixed --- Another B-----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB_another = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=my_base_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB_another.do_tune_kernelparams_negmllk(batch_size_tune=5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True) # bs 5; lr 0.2; epochs 5
    torch.manual_seed(0)
    #  set B
    Ty2_SCV_vectorvaluedfunc_fixB_another.B = torch.Tensor([[0.5, 0.01], [0.01, 0.5]])   # a value close to estimated B
    Ty2_SCV_vectorvaluedfunc_fixB_another.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)  # 0.002 ; 5



    # ---------------
    # vv-CV: MD1P with learning B
    print("REP {} out of {} --- vv-CV: MD1P with learning B -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc = VV_CV_vectorvaluedfuncs_model(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc, prior_kernel=stein_base_kernel_MV_2, base_kernel=my_base_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc.do_tune_kernelparams_negmllk(batch_size_tune = 5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True) # bs 5; lr 0.2; epochs 5
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc.do_optimize_vv_CV(regularizer_const=1e-5, regularizer_const_FB=1, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)  # 0.002; 5




    # --------------
    # sv-polynomials: f1
    print("REP {} out of {} --- sv-polynomials: f1 -----------".format(i + 1, no_replica_ty2))
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f1 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, polynomial_kernel, X1, Y1, score_X1)
    Ty2_SCV_svpolynomials_f1.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f1.do_optimize_sv_CV(regularizer_const=1e-5, batch_size=10, lr=my_lr, epochs=no_epochs_ty2, verbose=True)  # 0.002


    # sv-polynomials: f2
    print("REP {} out of {} --- sv-polynomials: f2 -----------".format(i + 1, no_replica_ty2))
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f2 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, polynomial_kernel, X2, Y2, score_X2)
    Ty2_SCV_svpolynomials_f2.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f2.do_optimize_sv_CV(regularizer_const=1e-5, batch_size=10, lr=my_lr, epochs=no_epochs_ty2,  verbose=True)  # 0.002



    # vv-polynomials: MD1P with B fixed
    print("REP {} out of {} --- vv-polynomials: MD1P with B fixed -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P_fixB = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=polynomial_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    Ty2_SCV_vvpolynomials_MD1P_fixB.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    #  set B
    Ty2_SCV_vvpolynomials_MD1P_fixB.B =  torch.Tensor([[0.1, 0.01], [0.01,0.1]])
    Ty2_SCV_vvpolynomials_MD1P_fixB.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)



    # vv-polynomials: MD1P with B fixed --- ANOTHER B
    print("REP {} out of {} --- vv-polynomials: MD1P with B fixed ---Another B -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P_fixB_another = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=polynomial_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    Ty2_SCV_vvpolynomials_MD1P_fixB_another.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    # set B
    Ty2_SCV_vvpolynomials_MD1P_fixB_another.B = torch.Tensor([[0.5, 0.01], [0.01, 0.5]])
    Ty2_SCV_vvpolynomials_MD1P_fixB_another.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)



    # vv-polynomials: MD1P with learning B
    print("REP {} out of {} --- vv-polynomials: MD1P with learning B -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P = VV_CV_vectorvaluedfuncs_model(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc, prior_kernel=stein_base_kernel_MV_2, base_kernel=polynomial_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    Ty2_SCV_vvpolynomials_MD1P.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P.do_optimize_vv_CV(regularizer_const=1e-5, regularizer_const_FB=1, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)




# Define a helper function to caculate the density of some samples from a standard normal distributions
def helper_standard_Gaussian_PDF(x):
    assert x.size(1)==1, "Dim should be 1"

    n = x.size(0)
    d = x.size(1)
    prob_densities_at_x = torch.zeros(n)

    for i in range(n):
        cur_x = x[i].squeeze()
        prob_densities_at_x[i] =  ((2.*math.pi)**(-0.5)) * torch.exp(-0.5* (cur_x.pow(2)) )

    return prob_densities_at_x





## Plot a fitted line for squared exponetial kernel.
sns.set_style("white")
all_x = torch.cat((X1, X2), dim=0)
all_x_dens = helper_standard_Gaussian_PDF(all_x)
all_x = all_x.squeeze()
all_x.size()


X1_sorted_values, X1_sorted_indices = X1.squeeze().sort()
X2_sorted_values, X2_sorted_indices = X2.squeeze().sort()

test_x = torch.unique(torch.sort(torch.cat((X1_sorted_values, X2_sorted_values, torch.linspace(-3, 3, 100)))).values)
test_x = test_x.unsqueeze(1)
test_x.size()

test_x_sorted_values, test_x_sorted_indices = test_x.squeeze().sort()


score_X1 = multivariate_Normal_score(mu, cov, X1)
score_X2 = multivariate_Normal_score(mu, cov, X2)
score_all_x = multivariate_Normal_score(mu, cov, all_x.unsqueeze(1))
score_test_x = multivariate_Normal_score(mu, cov, test_x )

vv_SEk_theta_hat = Ty2_SCV_vectorvaluedfunc.fitting_obj.theta.detach().clone()
vv_SEk_B  = Ty2_SCV_vectorvaluedfunc.fitting_obj.B.detach().clone()
vv_SEk_est  =  Ty2_SCV_vectorvaluedfunc.fitting_obj.c.detach().clone().squeeze()
with torch.no_grad():
    vv_SEk_k_XX = Ty2_SCV_vectorvaluedfunc.fitting_obj.kernel_obj.cal_stein_base_kernel(test_x, all_x.unsqueeze(1), score_test_x, score_all_x)
    vv_SEk_y_fitted = vv_SEk_k_XX @ vv_SEk_theta_hat @ vv_SEk_B + vv_SEk_est
vv_SEk_y_fitted.size()
vv_SEk_data_sorted_values, vv_SEk_data_sorted_indices = all_x.sort()




vv_1polnk_theta_hat = Ty2_SCV_vvpolynomials_MD1P.fitting_obj.theta.detach().clone()
vv_1polnk_B  = Ty2_SCV_vvpolynomials_MD1P.fitting_obj.B.detach().clone()
vv_1polnk_est  =  Ty2_SCV_vvpolynomials_MD1P.fitting_obj.c.detach().clone().squeeze()
with torch.no_grad():
    vv_1polnk_k_XX = Ty2_SCV_vvpolynomials_MD1P.fitting_obj.kernel_obj.cal_stein_base_kernel(test_x, all_x.unsqueeze(1), score_test_x, score_all_x)
    vv_1polnk_y_fitted = vv_1polnk_k_XX @ vv_1polnk_theta_hat @ vv_1polnk_B + vv_1polnk_est
vv_1polnk_y_fitted.size()
vv_1polnk_data_sorted_values, vv_1polnk_data_sorted_indices = all_x.sort()




sv_SEk_LF_theta_hat = Ty2_SCV_scalarvaluedfunc1.fitting_obj.theta.detach().clone()
sv_SEk_LF_est = Ty2_SCV_scalarvaluedfunc1.fitting_obj.c.clone().detach()
with torch.no_grad():
    sv_SEk_LF_k_XX = Ty2_SCV_scalarvaluedfunc1.fitting_obj.kernel_obj.cal_stein_base_kernel(test_x, X1, score_test_x,  score_X1)
    sv_SEk_LF_y_fitted = sv_SEk_LF_k_XX @ sv_SEk_LF_theta_hat + sv_SEk_LF_est
sv_SEk_LF_y_fitted = sv_SEk_LF_y_fitted.squeeze()
sv_SEk_LF_data_sorted_values, sv_SEk_LF_data_sorted_indices = X1.squeeze().sort()


sv_SEk_HF_theta_hat = Ty2_SCV_scalarvaluedfunc2.fitting_obj.theta.detach().clone()
sv_SEk_HF_est = Ty2_SCV_scalarvaluedfunc2.fitting_obj.c.clone().detach()
with torch.no_grad():
    sv_SEk_HF_k_XX  = Ty2_SCV_scalarvaluedfunc2.fitting_obj.kernel_obj.cal_stein_base_kernel(test_x, X2, score_test_x,  score_X2)
    sv_SEk_HF_y_fitted = sv_SEk_HF_k_XX @ sv_SEk_HF_theta_hat + sv_SEk_HF_est
sv_SEk_HF_y_fitted = sv_SEk_HF_y_fitted.squeeze()
sv_SEk_HF_data_sorted_values, sv_SEk_HF_data_sorted_indices = X2.squeeze().sort()



x_step = np.linspace(-3,3, 3)
y_LF = [-1, -1, 2]
y_HF = [0,  0 , 1]
x_illu = np.linspace(-3, 3, 500)


# Extract Saved Outputs
with open('../Step_funcion_all_data.pkl', 'rb') as input:
    no_replica_ty2 = pickle.load(input)

    no_epochs_ty2 = pickle.load(input)

    no_points_per_func_ty2 = pickle.load(input)

    #
    large_saved_MC_ests_ty2 = pickle.load(input)

    large_save_est_scalar_f1_ty2 = pickle.load(input)

    large_save_closed_form_sols_scalar_f1_ty2 = pickle.load(input)

    large_save_est_scalar_f2_ty2 = pickle.load(input)

    large_save_closed_form_sols_scalar_f2_ty2 = pickle.load(input)

    large_save_est_vecfunc_ty2 = pickle.load(input)

    large_save_est_vecfunc_fixB_ty2 = pickle.load(input)

    large_save_est_vecfunc_fixB_another_ty2 = pickle.load(input)


    # sv-polynomials
    large_save_est_scalar_f1_svpolynomials_ty2 = pickle.load(input)

    large_save_closed_form_sols_scalar_f1_svpolynomials_ty2 = pickle.load(input)

    large_save_est_scalar_f2_svpolynomials_ty2 = pickle.load(input)

    large_save_closed_form_sols_scalar_f2_svpolynomials_ty2 = pickle.load(input)

    large_save_est_vecfunc_vvpolynomials_ty2_MD1P = pickle.load(input)

    large_save_est_vecfunc_vvpolynomials_fixB_ty2 = pickle.load(input)

    large_save_est_vecfunc_vvpolynomials_fixB_another_ty2 = pickle.load(input)





with torch.no_grad():
    true_vals = [0.5, 0.5]
    fig, axs = plt.subplots(1, 4, sharex=False, sharey=False)
    fig.set_figwidth(20)
    sns.set_style("ticks")  # sns.set_style("whitegrid")

    clrs = sns.color_palette("husl", 16)
    start_pos = 0
    axs[2].set_xlabel('Number of Epochs', fontsize=20)
    axs[3].set_xlabel('Number of Epochs', fontsize=20)
    axs[2].tick_params(labelsize=20)
    axs[3].tick_params(labelsize=20)

    show_indx = np.arange(0, 410, 20)
    show_indx = show_indx - 1
    show_indx[0] = 0
    show_indx

    axs[2].set_title("Squared-exponential kernel CVs", fontsize=18)
    axs[3].set_title("First-order polynomial kernel CVs", fontsize=18)
    axs[2].set_ylabel(r'Absolute error for $\Pi_H [f_H]$', fontsize=18)

    # fig.set_figwidth(12)
    mc_f1_mean_ty2 = (large_saved_MC_ests_ty2[:, 0] - true_vals[0]).abs().mean().repeat(1, no_epochs_ty2)
    mc_f2_mean_ty2 = (large_saved_MC_ests_ty2[:, 1] - true_vals[1]).abs().mean().repeat(1, no_epochs_ty2)
    mc_f1_std_ty2 = (large_saved_MC_ests_ty2[:, 0] - true_vals[0]).abs().std(dim=0) / (torch.ones(1) * no_replica_ty2).sqrt().repeat(1, no_epochs_ty2)
    mc_f2_std_ty2 = (large_saved_MC_ests_ty2[:, 1] - true_vals[1]).abs().std(dim=0) / (torch.ones(1) * no_replica_ty2).sqrt().repeat(1, no_epochs_ty2)

    axs[2].axhline(mc_f2_mean_ty2[0, 0], color='black', label='MC')
    axs[3].axhline(mc_f2_mean_ty2[0, 0], color='black', label='MC')

    axs[2].axhline((large_save_closed_form_sols_scalar_f2_ty2 - true_vals[1]).abs().mean().detach().numpy(), color='black', linestyle='-.', label='CF')
    axs[3].axhline((large_save_closed_form_sols_scalar_f2_svpolynomials_ty2 - true_vals[1]).abs().mean().detach().numpy(),color='black', linestyle='-.', label='CF')

    # -------
    sv_f1_mean_ty2 = (large_save_est_scalar_f1_ty2 - true_vals[0]).abs().mean(dim=0).detach().numpy()
    sv_f2_mean_ty2 = (large_save_est_scalar_f2_ty2 - true_vals[1]).abs().mean(dim=0).detach().numpy()
    sv_f1_std_ty2 = (large_save_est_scalar_f1_ty2 - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    sv_f2_std_ty2 = (large_save_est_scalar_f2_ty2 - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[2].plot(show_indx + 1, sv_f2_mean_ty2[show_indx], c=clrs[1], marker='+', label='CV')
    axs[2].fill_between(show_indx + 1, sv_f2_mean_ty2[show_indx] - sv_f2_std_ty2[show_indx], sv_f2_mean_ty2[show_indx] + sv_f2_std_ty2[show_indx], alpha=0.3, facecolor=clrs[1])

    # -------
    vv_f1_mean_ty2_fixB = (large_save_est_vecfunc_fixB_ty2[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
    vv_f2_mean_ty2_fixB = (large_save_est_vecfunc_fixB_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
    vv_f1_std_ty2_fixB = (large_save_est_vecfunc_fixB_ty2[:, :, 0] - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    vv_f2_std_ty2_fixB = (large_save_est_vecfunc_fixB_ty2[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[2].plot(show_indx + 1, (large_save_est_vecfunc_fixB_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[show_indx],\
                c=clrs[7], marker='x', label='vv-CV with Fixed B (1)')
    axs[2].fill_between(show_indx + 1, vv_f2_mean_ty2_fixB[show_indx] - vv_f2_std_ty2_fixB[show_indx], vv_f2_mean_ty2_fixB[show_indx] + vv_f2_std_ty2_fixB[show_indx], alpha=0.3, facecolor=clrs[7])

    # -------
    vv_f1_mean_ty2_fixB_another = (large_save_est_vecfunc_fixB_another_ty2[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
    vv_f2_mean_ty2_fixB_another = (large_save_est_vecfunc_fixB_another_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
    vv_f1_std_ty2_fixB_another = (large_save_est_vecfunc_fixB_another_ty2[:, :, 0] - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    vv_f2_std_ty2_fixB_another = (large_save_est_vecfunc_fixB_another_ty2[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[2].plot(show_indx + 1,(large_save_est_vecfunc_fixB_another_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[ show_indx], c=clrs[3], marker='x', label='vv-CV with Fixed B (2)')
    axs[2].fill_between(show_indx + 1, vv_f2_mean_ty2_fixB_another[show_indx] - vv_f2_std_ty2_fixB_another[show_indx],vv_f2_mean_ty2_fixB_another[show_indx] + vv_f2_std_ty2_fixB_another[show_indx], alpha=0.3, facecolor=clrs[5])

    # -------
    vv_f1_mean_ty2 = (large_save_est_vecfunc_ty2[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
    vv_f2_mean_ty2 = (large_save_est_vecfunc_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
    vv_f1_std_ty2 = (large_save_est_vecfunc_ty2[:, :, 0] - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    vv_f2_std_ty2 = (large_save_est_vecfunc_ty2[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[2].plot(show_indx + 1,(large_save_est_vecfunc_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[show_indx], c=clrs[10], marker='.', label='vv-CV with Estimated B')
    axs[2].fill_between(show_indx + 1, vv_f2_mean_ty2[show_indx] - vv_f2_std_ty2[show_indx], vv_f2_mean_ty2[show_indx] + vv_f2_std_ty2[show_indx], alpha=0.3, facecolor=clrs[10])

    # -------
    svpoly_f1_mean_ty2 = (large_save_est_scalar_f1_svpolynomials_ty2 - true_vals[0]).abs().mean(dim=0).detach().numpy()
    svpoly_f2_mean_ty2 = (large_save_est_scalar_f2_svpolynomials_ty2 - true_vals[1]).abs().mean(dim=0).detach().numpy()
    svpoly_f1_std_ty2 = (large_save_est_scalar_f1_svpolynomials_ty2 - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    svpoly_f2_std_ty2 = (large_save_est_scalar_f2_svpolynomials_ty2 - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[3].plot(show_indx + 1, svpoly_f2_mean_ty2[show_indx], c=clrs[1], marker='+', label='CV')
    axs[3].fill_between(show_indx + 1, svpoly_f2_mean_ty2[show_indx] - svpoly_f2_std_ty2[show_indx], svpoly_f2_mean_ty2[show_indx] + svpoly_f2_std_ty2[show_indx], alpha=0.3, facecolor=clrs[1])

    # -------
    vvpoly_f1_mean_ty2_fixB = (large_save_est_vecfunc_vvpolynomials_fixB_ty2[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
    vvpoly_f2_mean_ty2_fixB = (large_save_est_vecfunc_vvpolynomials_fixB_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
    vvpoly_f1_std_ty2_fixB = (large_save_est_vecfunc_vvpolynomials_fixB_ty2[:, :, 0] - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    vvpoly_f2_std_ty2_fixB = (large_save_est_vecfunc_vvpolynomials_fixB_ty2[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[3].plot(show_indx + 1, (large_save_est_vecfunc_vvpolynomials_fixB_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[show_indx], c=clrs[7], marker='x', label='vv-CV with Fixed B (1)')
    axs[3].fill_between(show_indx + 1, vvpoly_f2_mean_ty2_fixB[show_indx] - vvpoly_f2_std_ty2_fixB[show_indx], vvpoly_f2_mean_ty2_fixB[show_indx] + vvpoly_f2_std_ty2_fixB[show_indx], alpha=0.3, facecolor=clrs[7])

    # -------
    vvpoly_f1_mean_ty2_fixB_another = (large_save_est_vecfunc_vvpolynomials_fixB_another_ty2[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
    vvpoly_f2_mean_ty2_fixB_another = (large_save_est_vecfunc_vvpolynomials_fixB_another_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
    vvpoly_f1_std_ty2_fixB_another = (large_save_est_vecfunc_vvpolynomials_fixB_another_ty2[:, :, 0] - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    vvpoly_f2_std_ty2_fixB_another = (large_save_est_vecfunc_vvpolynomials_fixB_another_ty2[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[3].plot(show_indx + 1, (large_save_est_vecfunc_vvpolynomials_fixB_another_ty2[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[show_indx], c=clrs[3], marker='x', label='vv-CV with Fixed B (2)')
    axs[3].fill_between(show_indx + 1,vvpoly_f2_mean_ty2_fixB_another[show_indx] - vvpoly_f2_std_ty2_fixB_another[show_indx], vvpoly_f2_mean_ty2_fixB_another[show_indx] + vvpoly_f2_std_ty2_fixB_another[show_indx],alpha=0.3, facecolor=clrs[5])

    # -------
    vvpoly_f1_mean_ty2 = (large_save_est_vecfunc_vvpolynomials_ty2_MD1P[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
    vvpoly_f2_mean_ty2 = (large_save_est_vecfunc_vvpolynomials_ty2_MD1P[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
    vvpoly_f1_std_ty2 = (large_save_est_vecfunc_vvpolynomials_ty2_MD1P[:, :, 0] - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    vvpoly_f2_std_ty2 = (large_save_est_vecfunc_vvpolynomials_ty2_MD1P[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica_ty2)
    axs[3].plot(show_indx + 1, (large_save_est_vecfunc_vvpolynomials_ty2_MD1P[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[show_indx], c=clrs[10], marker='.', label='vv-CV with Estimated B')
    axs[3].fill_between(show_indx + 1, vvpoly_f2_mean_ty2[show_indx] - vvpoly_f2_std_ty2[show_indx], vvpoly_f2_mean_ty2[show_indx] + vvpoly_f2_std_ty2[show_indx], alpha=0.3, facecolor=clrs[10])

    # If want to include the legend inside the figure
    axs[2].legend(loc="upper right", fontsize=13)

    # sns.set_style("ticks") # sns.set_style("whitegrid")
    axs[0].set_title("Low-fidelity model", fontsize=18)
    axs[1].set_title("High-fidelity model", fontsize=18)

    axs[0].set_ylim([-3, 3])
    axs[1].set_ylim([-3, 3])
    axs[2].set_ylim([0.03, 0.07])
    axs[3].set_ylim([0.03, 0.07])

    axs[0].plot(test_x_sorted_values, vv_SEk_y_fitted[:, 0][test_x_sorted_indices], color='blue', ls='dotted',label='vv-CV')
    axs[0].plot(test_x_sorted_values, vv_1polnk_y_fitted[:, 0][test_x_sorted_indices], color='orange', ls='dotted',label='vv-CV (1st order polyn. k)')
    axs[0].plot(test_x_sorted_values, sv_SEk_LF_y_fitted[test_x_sorted_indices], color='red', ls='dotted',label='CV (squared-exponetial k)')

    axs[0].step(x_step, y_LF, color='black', label=r'$f(x)$')
    axs[1].set_xlabel("x", fontsize=20)
    axs[1].set_ylabel("y", fontsize=20)
    axs[1].tick_params(labelsize=20)

    axs[1].plot(test_x_sorted_values, vv_SEk_y_fitted[:, 1][test_x_sorted_indices], color='blue', ls='dotted',label='vv-CV (squared-exponetial k)')
    axs[1].plot(test_x_sorted_values, vv_1polnk_y_fitted[:, 1][test_x_sorted_indices], color='orange', ls='dotted', label='vv-CV (1st order polyn. k)')
    axs[1].plot(test_x_sorted_values, sv_SEk_HF_y_fitted[test_x_sorted_indices], color='red', ls='dotted', label='CV (squared-exponetial k)')

    axs[1].step(x_step, y_HF, color='black', label=r'$f(x)$')
    axs[0].set_xlabel("x", fontsize=20)
    axs[0].set_ylabel("y", fontsize=20)
    axs[0].tick_params(labelsize=20)

    axs[1].legend(fontsize=13)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=15)

    plt.show()

    fig.savefig('step_function_plot.pdf')







