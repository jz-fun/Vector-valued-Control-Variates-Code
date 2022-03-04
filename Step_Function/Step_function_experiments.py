import pickle
from src.src_vvCV_MD1P.stein_operators import *
from src.src_vvCV_MD1P.sv_CV import *
from src.src_vvCV_MD1P.vv_CV_MD1P import *
from src.src_vvCV_MD1P.vv_CV_FixB_MD1P import *
from src.src_vvCV_MD1P.vv_CV_unbalanced_FixB_MD1P import *



# ==========================
# Step function experiments
# ==========================
# Set  vv-CV kernel
my_base_kernel = rbf_kernel

my_lr = 0.0003

my_poly_ker_parm = torch.Tensor([1,1])

no_replica_ty2 = 100
no_epochs_ty2 = 400
no_points_per_func_ty2= 40

# Store MC ests
large_saved_MC_ests_ty2 = torch.zeros(no_replica_ty2, 2)

# Store sv-CVs
large_save_est_scalar_f1_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2)
large_save_closed_form_sols_scalar_f1_ty2 = torch.zeros(no_replica_ty2)

large_save_est_scalar_f2_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2)
large_save_closed_form_sols_scalar_f2_ty2 = torch.zeros(no_replica_ty2)


# Store vv-CV-MD1P
large_save_est_vecfunc_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2, 2)


# Store vv-CV-MD1P with B fixed
large_save_est_vecfunc_fixB_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2, 2)

large_save_est_vecfunc_fixB_another_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2, 2)




#---
# Store sv-polynomials-CVs
large_save_est_scalar_f1_svpolynomials_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2)
large_save_closed_form_sols_scalar_f1_svpolynomials_ty2 = torch.zeros(no_replica_ty2)

large_save_est_scalar_f2_svpolynomials_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2)
large_save_closed_form_sols_scalar_f2_svpolynomials_ty2 = torch.zeros(no_replica_ty2)


# Store vv-polynomials MD1P
large_save_est_vecfunc_vvpolynomials_ty2_MD1P = torch.zeros(no_replica_ty2, no_epochs_ty2, 2)

# Store vv-polynomials MD1P with B fixed
large_save_est_vecfunc_vvpolynomials_fixB_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2, 2)

large_save_est_vecfunc_vvpolynomials_fixB_another_ty2 = torch.zeros(no_replica_ty2, no_epochs_ty2, 2)





for i in range(no_replica_ty2):
    print("REP {} out of {}-----------".format(i+1, no_replica_ty2 ))
    dim = 1
    factor = torch.ones(1) * 1
    mu = torch.zeros(dim, dtype=torch.float) + 0
    var = torch.eye(dim, dtype=torch.float) * factor
    print(mu, var)


    def my_func_1(X):
        return (0.5 + (2 * (X >= 0) - 1) * 1.5) * torch.ones(1, dtype=torch.float)


    def my_func_2(X):
        return (X >= 0) * torch.ones(1, dtype=torch.float)



    # Training samples
    print("REP {} out of {}-----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(2*(i+2))
    X1 = mu + torch.sqrt(factor) * torch.randn(no_points_per_func_ty2, dim)
    Y1 = my_func_1(X1)


    # --- For MD1P
    torch.manual_seed( 2* (i+2) + 1)
    X2 = mu + torch.sqrt(factor) * torch.randn(no_points_per_func_ty2, dim)
    Y2 = my_func_2(X2)


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

    # Monte Carlo estimates
    large_saved_MC_ests_ty2[i] = torch.Tensor([Y1.mean(dim=0) , Y2.mean(dim=0)])


    # f1
    print("REP {} out of {} --- sv-CV-f1 -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc1 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, my_base_kernel, X1, Y1, score_X1)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc1.do_tune_kernelparams_negmllk(batch_size_tune = 5, flag_if_use_medianheuristic=False,  beta_cstkernel=0., lr=0.02, epochs=15, verbose=True)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc1.do_optimize_sv_CV(regularizer_const = 1e-5, batch_size = 10, lr = my_lr, epochs = no_epochs_ty2, verbose = True)
    large_save_est_scalar_f1_ty2[i] = Ty2_SCV_scalarvaluedfunc1.saved_BQ_est
    large_save_closed_form_sols_scalar_f1_ty2[i] = Ty2_SCV_scalarvaluedfunc1.do_closed_form_est_for_simpliedCF().squeeze().detach().clone()


    # f2
    print("REP {} out of {}--- sv-CV-f2 -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc2 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc,stein_base_kernel_MV_2, my_base_kernel, X2, Y2, score_X2)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc2.do_tune_kernelparams_negmllk(batch_size_tune = 5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True)
    torch.manual_seed(0)
    Ty2_SCV_scalarvaluedfunc2.do_optimize_sv_CV(regularizer_const=1e-5, batch_size=10, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_scalar_f2_ty2[i] = Ty2_SCV_scalarvaluedfunc2.saved_BQ_est
    large_save_closed_form_sols_scalar_f2_ty2[i] = Ty2_SCV_scalarvaluedfunc2.do_closed_form_est_for_simpliedCF().squeeze().detach().clone()



    # vv-CV: MD1P with B fixed
    print("REP {} out of {} --- vv-CV: MD1P with B fixed -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=my_base_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB.do_tune_kernelparams_negmllk(batch_size_tune=5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB.B =  torch.Tensor([[0.1, 0.01], [0.01,0.1]])
    Ty2_SCV_vectorvaluedfunc_fixB.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_vecfunc_fixB_ty2[i] = Ty2_SCV_vectorvaluedfunc_fixB.saved_BQ_est.squeeze()



    # vv-CV: MD1P with B fixed --  ANOTHER B
    print("REP {} out of {} --- vv-CV: MD1P with B fixed --- Another B-----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB_another = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=my_base_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB_another.do_tune_kernelparams_negmllk(batch_size_tune=5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc_fixB_another.B = torch.Tensor([[0.5, 0.01], [0.01, 0.5]])
    Ty2_SCV_vectorvaluedfunc_fixB_another.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_vecfunc_fixB_another_ty2[i] = Ty2_SCV_vectorvaluedfunc_fixB_another.saved_BQ_est.squeeze()



    # vv-CV: MD1P with learning B
    print("REP {} out of {} --- vv-CV: MD1P with learning B -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc = VV_CV_vectorvaluedfuncs_model(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc, prior_kernel=stein_base_kernel_MV_2, base_kernel=my_base_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc.do_tune_kernelparams_negmllk(batch_size_tune = 5, flag_if_use_medianheuristic=False, beta_cstkernel=0., lr=0.02, epochs=15, verbose=True)
    torch.manual_seed(0)
    Ty2_SCV_vectorvaluedfunc.do_optimize_vv_CV(regularizer_const=1e-5, regularizer_const_FB=1, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_vecfunc_ty2[i] = Ty2_SCV_vectorvaluedfunc.saved_BQ_est.squeeze()




    # --------------
    # sv-polynomials: f1
    print("REP {} out of {} --- sv-polynomials: f1 -----------".format(i + 1, no_replica_ty2))
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f1 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, polynomial_kernel, X1, Y1, score_X1)
    Ty2_SCV_svpolynomials_f1.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f1.do_optimize_sv_CV(regularizer_const=1e-5, batch_size=10, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_scalar_f1_svpolynomials_ty2[i] = Ty2_SCV_svpolynomials_f1.saved_BQ_est
    large_save_closed_form_sols_scalar_f1_svpolynomials_ty2[i] =Ty2_SCV_svpolynomials_f1.do_closed_form_est_for_simpliedCF().squeeze().detach().clone()


    # sv-polynomials: f2
    print("REP {} out of {} --- sv-polynomials: f2 -----------".format(i + 1, no_replica_ty2))
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f2 = SV_CV_scalarvaluedfuncs_model(penalized_ls_objective_scalarvaluedfunc, stein_base_kernel_MV_2, polynomial_kernel, X2, Y2, score_X2)
    Ty2_SCV_svpolynomials_f2.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_svpolynomials_f2.do_optimize_sv_CV(regularizer_const=1e-5, batch_size=10, lr=my_lr, epochs=no_epochs_ty2,  verbose=True)
    large_save_est_scalar_f2_svpolynomials_ty2[i] = Ty2_SCV_svpolynomials_f2.saved_BQ_est
    large_save_closed_form_sols_scalar_f2_svpolynomials_ty2[i] = Ty2_SCV_svpolynomials_f2.do_closed_form_est_for_simpliedCF().squeeze().detach().clone()




    # vv-polynomials: MD1P with B fixed
    print("REP {} out of {} --- vv-polynomials: MD1P with B fixed -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P_fixB = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=polynomial_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    Ty2_SCV_vvpolynomials_MD1P_fixB.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P_fixB.B =  torch.Tensor([[0.1, 0.01], [0.01,0.1]])
    Ty2_SCV_vvpolynomials_MD1P_fixB.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_vecfunc_vvpolynomials_fixB_ty2[i] = Ty2_SCV_vvpolynomials_MD1P_fixB.saved_BQ_est.squeeze()





    # vv-polynomials: MD1P with B fixed --- ANOTHER B
    print("REP {} out of {} --- vv-polynomials: MD1P with B fixed ---Another B -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P_fixB_another = VV_CV_vectorvaluedfuncs_model_fixB(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB, prior_kernel=stein_base_kernel_MV_2, base_kernel=polynomial_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    Ty2_SCV_vvpolynomials_MD1P_fixB_another.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P_fixB_another.B = torch.Tensor([[0.5, 0.01], [0.01, 0.5]])
    Ty2_SCV_vvpolynomials_MD1P_fixB_another.do_optimize_vv_CV(regularizer_const=1e-5, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_vecfunc_vvpolynomials_fixB_another_ty2[i] = Ty2_SCV_vvpolynomials_MD1P_fixB_another.saved_BQ_est.squeeze()



    # vv-polynomials: MD1P with learning B
    print("REP {} out of {} --- vv-polynomials: MD1P with learning B -----------".format(i+1, no_replica_ty2 ))
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P = VV_CV_vectorvaluedfuncs_model(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc, prior_kernel=stein_base_kernel_MV_2, base_kernel=polynomial_kernel, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
    Ty2_SCV_vvpolynomials_MD1P.optim_base_kernel_parms = my_poly_ker_parm
    torch.manual_seed(0)
    Ty2_SCV_vvpolynomials_MD1P.do_optimize_vv_CV(regularizer_const=1e-5, regularizer_const_FB=1, batch_size=5, lr=my_lr, epochs=no_epochs_ty2, verbose=True)
    large_save_est_vecfunc_vvpolynomials_ty2_MD1P[i] = Ty2_SCV_vvpolynomials_MD1P.saved_BQ_est.squeeze()






##################
# Save Outputs
##################
# Comments: If you want to rerun the above experiment and save your own results, please uncomment the following block to save your data.
#
# with open('../data/Step_funcion_all_data.pkl', 'wb') as output:
#     #
#     no_replica_ty2 = no_replica_ty2
#     pickle.dump(no_replica_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#     no_epochs_ty2 = no_epochs_ty2
#     pickle.dump(no_epochs_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#     no_points_per_func_ty2 = no_points_per_func_ty2
#     pickle.dump(no_points_per_func_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     # MC
#     large_saved_MC_ests_ty2 = large_saved_MC_ests_ty2
#     pickle.dump(large_saved_MC_ests_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#     # sv-CVs
#     large_save_est_scalar_f1_ty2 = large_save_est_scalar_f1_ty2
#     pickle.dump(large_save_est_scalar_f1_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#     large_save_closed_form_sols_scalar_f1_ty2 = large_save_closed_form_sols_scalar_f1_ty2
#     pickle.dump(large_save_closed_form_sols_scalar_f1_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     large_save_est_scalar_f2_ty2 = large_save_est_scalar_f2_ty2
#     pickle.dump(large_save_est_scalar_f2_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#     large_save_closed_form_sols_scalar_f2_ty2 = large_save_closed_form_sols_scalar_f2_ty2
#     pickle.dump(large_save_closed_form_sols_scalar_f2_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     # vv-CV-MD1P
#     large_save_est_vecfunc_ty2 = large_save_est_vecfunc_ty2
#     pickle.dump(large_save_est_vecfunc_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     #  vv-CV-MD1P-fixed B
#     large_save_est_vecfunc_fixB_ty2 = large_save_est_vecfunc_fixB_ty2
#     pickle.dump(large_save_est_vecfunc_fixB_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     #  vv-CV-MD1P-fixed B (Another)
#     large_save_est_vecfunc_fixB_another_ty2 = large_save_est_vecfunc_fixB_another_ty2
#     pickle.dump(large_save_est_vecfunc_fixB_another_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#
#     ## sv/vv-polynomials
#     #
#     large_save_est_scalar_f1_svpolynomials_ty2 = large_save_est_scalar_f1_svpolynomials_ty2
#     pickle.dump(large_save_est_scalar_f1_svpolynomials_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#     large_save_closed_form_sols_scalar_f1_svpolynomials_ty2 = large_save_closed_form_sols_scalar_f1_svpolynomials_ty2
#     pickle.dump(large_save_closed_form_sols_scalar_f1_svpolynomials_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     large_save_est_scalar_f2_svpolynomials_ty2 = large_save_est_scalar_f2_svpolynomials_ty2
#     pickle.dump(large_save_est_scalar_f2_svpolynomials_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     large_save_closed_form_sols_scalar_f2_svpolynomials_ty2 = large_save_closed_form_sols_scalar_f2_svpolynomials_ty2
#     pickle.dump(large_save_closed_form_sols_scalar_f2_svpolynomials_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     #
#     large_save_est_vecfunc_vvpolynomials_ty2_MD1P = large_save_est_vecfunc_vvpolynomials_ty2_MD1P
#     pickle.dump(large_save_est_vecfunc_vvpolynomials_ty2_MD1P, output, pickle.HIGHEST_PROTOCOL)
#
#     #
#     large_save_est_vecfunc_vvpolynomials_fixB_ty2 = large_save_est_vecfunc_vvpolynomials_fixB_ty2
#     pickle.dump(large_save_est_vecfunc_vvpolynomials_fixB_ty2, output, pickle.HIGHEST_PROTOCOL)
#
#
#     #
#     large_save_est_vecfunc_vvpolynomials_fixB_another_ty2 = large_save_est_vecfunc_vvpolynomials_fixB_another_ty2
#     pickle.dump(large_save_est_vecfunc_vvpolynomials_fixB_another_ty2, output, pickle.HIGHEST_PROTOCOL)







