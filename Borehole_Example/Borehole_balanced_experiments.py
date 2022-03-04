import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Borehole_Example.src_ProdKernel_vvCV_MD1P.vv_CV_MD1P_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.vv_CV_FixB_MD1P_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.stein_operators_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.product_base_kernels_borehole import *




def my_func_1(X):
    # f1 is the low fidelity model
    assert X.dim() == 2, "Design points matrix X should be a 2d tensor; each row represents a instance. And each instance should match the dim of Borehole func."

    n = X.size()[0]
    d = X.size()[1]

    o = torch.zeros(n)
    for i in range(n):
        r_w  = X[i,0]
        r    = X[i,1]
        T_u  = X[i,2]
        T_l  = X[i,3]
        H_u  = X[i,4]
        H_l  = X[i,5]
        L    = X[i,6]
        K_w  = X[i,7]
        o[i] = 5*T_u*(H_u - H_l)/(torch.log(r/r_w) * (1.5 + (2 *L*T_u)/(torch.log(r/r_w) * (r_w ** 2) * K_w) + T_u / T_l))

    if o.dim() == 1:
        o = o.unsqueeze(dim=1)

    assert o.size() == torch.Size([n, 1])
    return o


def my_func_2(X):
    # f2 is the high fidelity model
    assert X.dim() == 2, "Design points matrix X should be a 2d tensor; each row represents a instance. And each instance should match the dim of Borehole func."

    n = X.size()[0]
    d = X.size()[1]

    o = torch.zeros(n)
    for i in range(n):
        r_w = X[i, 0]
        r   = X[i, 1]
        T_u = X[i, 2]
        T_l = X[i, 3]
        H_u = X[i, 4]
        H_l = X[i, 5]
        L   = X[i, 6]
        K_w = X[i, 7]
        o[i] =  2 * math.pi * T_u * (H_u - H_l) / (torch.log(r/r_w) * (1.0 + (2*L*T_u )/( torch.log(r/r_w)  * (r_w**2) * K_w) + T_u/T_l) )

    if o.dim() == 1:
        o = o.unsqueeze(dim=1)

    assert o.size() == torch.Size([n, 1])
    return o









def EXP_borehole_balanced(no_replica=20, no_epochs=800, no_points_per_func=50, my_batch_size_tune=5, my_lr_tune=0.2, my_tune_epochs=10, my_tune_verbose=True, my_regularizer_const_weights_optimCV=1e-5, my_batch_size_optimCV = 10, my_lr_optimCV=0.004, my_optimCV_verbose=True):
    NO_tasks= 2

    large_saved_MC_ests = torch.zeros(no_replica, NO_tasks)
    large_save_est_scalar_f1 = torch.zeros(no_replica, no_epochs)
    large_save_est_scalar_f2 = torch.zeros(no_replica, no_epochs)
    large_save_est_vecfunc = torch.zeros(no_replica, no_epochs, NO_tasks)
    large_save_closed_form_sols = torch.zeros(no_replica, NO_tasks)

    large_save_est_vecfunc_fixB = torch.zeros(no_replica, no_epochs, NO_tasks)
    large_save_closed_form_sols_fixB = torch.zeros(no_replica, NO_tasks)

    for i in range(no_replica):
        # Setting means and vars
        mu_r_w = torch.ones(1) * 0.1
        mu_r = torch.ones(1) * 100.  # 1.
        mu_T_u = torch.ones(1) * ((63070 + 115600) / 2)
        mu_T_l = torch.ones(1) * ((63.1 + 116) / 2)
        mu_H_u = torch.ones(1) * ((990 + 1110) / 2)
        mu_H_l = torch.ones(1) * ((700 + 820) / 2)
        mu_L = torch.ones(1) * ((1120 + 1680) / 2)
        mu_K_w = torch.ones(1) * ((9855 + 12045) / 2)

        var_r_w = torch.ones(1) * 0.0161812 ** 2
        var_r = torch.ones(1) * 0.01
        var_T_u = torch.ones(1) * 20.
        var_T_l = torch.ones(1) * 1.
        var_H_u = torch.ones(1) * 1.
        var_H_l = torch.ones(1) * 1.
        var_L = torch.ones(1) * 10.
        var_K_w = torch.ones(1) * 30.

        my_mus = torch.Tensor([mu_r_w, mu_r, mu_T_u, mu_T_l, mu_H_u, mu_H_l, mu_L, mu_K_w])
        my_mus = my_mus.unsqueeze(dim=1)  # has to be a col-vector
        my_mus.size()

        my_vars = torch.Tensor([var_r_w, var_r, var_T_u, var_T_l, var_H_u, var_H_l, var_L, var_K_w])
        my_vars = my_vars.unsqueeze(dim=1)  # has to be a col-vector
        my_vars.size()


        print("REP {} out of {}-----------".format(i + 1, no_replica))
        # Training samples
        m = no_points_per_func

        # Dataset for LF
        torch.manual_seed(2 * i)
        r_ws_X1 = mu_r_w + torch.sqrt(var_r_w) * torch.randn(m, 1)
        rs_X1 = mu_r + torch.sqrt(var_r) * torch.randn(m, 1)
        T_us_X1 = mu_T_u + torch.sqrt(var_T_u) * torch.randn(m, 1)
        T_ls_X1 = mu_T_l + torch.sqrt(var_T_l) * torch.randn(m, 1)
        H_us_X1 = mu_H_u + torch.sqrt(var_H_u) * torch.randn(m, 1)
        H_ls_X1 = mu_H_l + torch.sqrt(var_H_l) * torch.randn(m, 1)
        Ls_X1 = mu_L + torch.sqrt(var_L) * torch.randn(m, 1)
        K_ws_X1 = mu_K_w + torch.sqrt(var_K_w) * torch.randn(m, 1)

        X1 = torch.stack((r_ws_X1, rs_X1, T_us_X1, T_ls_X1, H_us_X1, H_ls_X1, Ls_X1, K_ws_X1), dim=1).squeeze()
        X1.size()
        Y1 = my_func_1(X1)
        Y1.size()


        # Dataset for HF
        torch.manual_seed(2 * i + 1)
        r_ws_X2 = mu_r_w + torch.sqrt(var_r_w) * torch.randn(m, 1)
        rs_X2 = mu_r + torch.sqrt(var_r) * torch.randn(m, 1)
        T_us_X2 = mu_T_u + torch.sqrt(var_T_u) * torch.randn(m, 1)
        T_ls_X2 = mu_T_l + torch.sqrt(var_T_l) * torch.randn(m, 1)
        H_us_X2 = mu_H_u + torch.sqrt(var_H_u) * torch.randn(m, 1)
        H_ls_X2 = mu_H_l + torch.sqrt(var_H_l) * torch.randn(m, 1)
        Ls_X2 = mu_L + torch.sqrt(var_L) * torch.randn(m, 1)
        K_ws_X2 = mu_K_w + torch.sqrt(var_K_w) * torch.randn(m, 1)

        X2 = torch.stack((r_ws_X2, rs_X2, T_us_X2, T_ls_X2, H_us_X2, H_ls_X2, Ls_X2, K_ws_X2), dim=1).squeeze()
        X2.size()
        Y2 = my_func_2(X2)
        Y2.size()

        # Compute scores
        score_X1 = product_Normal_score(my_mus, my_vars, X1)
        score_X1.size()
        score_X2 = product_Normal_score(my_mus, my_vars, X2)
        score_X2.size()

        xall = torch.stack((X1, X2), dim=0)
        xall.size()
        yall = torch.stack((Y1, Y2), dim=0)
        yall.size()
        score_all = torch.stack((score_X1, score_X2), dim=0)
        score_all.size()

        # Monte Carlo estimates
        large_saved_MC_ests[i] = torch.Tensor([Y1.mean(dim=0), Y2.mean(dim=0)])


        # vv-CV: MD1P with B fixed
        print("REP {} out of {} --- vv-CV: MD1P with B fixed -----------".format(i + 1, no_replica))
        my_SCV_vectorvaluedfunc_fixB = VV_CV_vectorvaluedfuncs_model_fixB_borehole(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_fixB_borehole,prior_kernel=stein_base_kernel_borehole, base_kernel=prod_rbf_kernel_Borehore, Xs_tensor=xall,Ys_tensor=yall, scores_Tensor=score_all)
        torch.manual_seed(0)
        my_SCV_vectorvaluedfunc_fixB.do_tune_kernelparams_negmllk(batch_size_tune=my_batch_size_tune, flag_if_use_medianheuristic=False, beta_cstkernel=1, lr=my_lr_tune, epochs=my_tune_epochs, verbose=my_tune_verbose)
        torch.manual_seed(0)
        #  Mannualy set a B
        my_SCV_vectorvaluedfunc_fixB.B = 0.005 * torch.Tensor([[0.1, 0.01], [0.01, 0.1]])
        my_SCV_vectorvaluedfunc_fixB.do_optimize_vv_CV(regularizer_const=my_regularizer_const_weights_optimCV, batch_size=np.int(my_batch_size_optimCV/NO_tasks), lr=my_lr_optimCV, epochs=no_epochs, verbose=my_optimCV_verbose)
        large_save_est_vecfunc_fixB[i] = my_SCV_vectorvaluedfunc_fixB.saved_BQ_est.squeeze().detach().clone()
        large_save_closed_form_sols_fixB[i] = torch.Tensor(my_SCV_vectorvaluedfunc_fixB.do_closed_form_est_for_simpliedCF()).squeeze().detach().clone()



        # vv-CV: MD1P with learning B
        print("REP {} out of {} --- vv-CV: MD1P with learning B -----------".format(i + 1, no_replica))
        torch.manual_seed(0)
        my_SCV_vectorvaluedfunc = VV_CV_vectorvaluedfuncs_model_borehole(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_borehole, prior_kernel=stein_base_kernel_borehole,base_kernel=prod_rbf_kernel_Borehore, Xs_tensor=xall, Ys_tensor=yall, scores_Tensor=score_all)
        my_SCV_vectorvaluedfunc.do_tune_kernelparams_negmllk(batch_size_tune=my_batch_size_tune, flag_if_use_medianheuristic=False, beta_cstkernel=1, lr=my_lr_tune, epochs=my_tune_epochs, verbose=my_tune_verbose)
        torch.manual_seed(0)
        my_SCV_vectorvaluedfunc.do_optimize_vv_CV(regularizer_const=my_regularizer_const_weights_optimCV, regularizer_const_FB=1, batch_size=np.int(my_batch_size_optimCV/NO_tasks), lr=my_lr_optimCV, epochs=no_epochs, verbose=my_optimCV_verbose)
        closed_form_sols1 = my_SCV_vectorvaluedfunc.do_closed_form_est_for_simpliedCF().detach().clone()
        large_save_closed_form_sols[i] = torch.Tensor(closed_form_sols1).squeeze()
        large_save_est_vecfunc[i] = my_SCV_vectorvaluedfunc.saved_BQ_est.squeeze().detach().clone()


    return  no_replica, no_epochs,  large_saved_MC_ests , large_save_est_scalar_f1, large_save_est_scalar_f2 , large_save_est_vecfunc_fixB , large_save_closed_form_sols_fixB , large_save_est_vecfunc ,large_save_closed_form_sols















###

class Borehole_balanced_exps(object):
    def __init__(self, set_of_ss, no_replica, no_epochs, set_of_batch_size_tune, set_of_lr_tune, set_of_tune_epochs, tune_verbose, set_of_regularizer_const_weights_optimCV, set_of_batch_size_optimCV, set_of_lr_optimCV, optimCV_verbose ):
        """
        :param set_of_ss: list
        :param no_replica: int
        :param no_epochs:  int
        :param set_of_batch_size_tune: list
        :param set_of_lr_tune:  list
        :param set_of_tune_epochs: list
        :param tune_verbose: bool
        :param set_of_regularizer_const_weights_optimCV: list
        :param set_of_batch_size_optimCV: list
        :param set_of_lr_optimCV: list
        :param optimCV_verbose: bool
        """


        self.set_of_ss = set_of_ss
        self.no_replica = no_replica
        self.no_epochs = no_epochs
        self.set_of_batch_size_tune = set_of_batch_size_tune
        self.set_of_lr_tune = set_of_lr_tune
        self.set_of_tune_epochs = set_of_tune_epochs
        self.tune_verbose = tune_verbose
        self.set_of_regularizer_const_weights_optimCV = set_of_regularizer_const_weights_optimCV
        self.set_of_batch_size_optimCV = set_of_batch_size_optimCV
        self.set_of_lr_optimCV = set_of_lr_optimCV
        self.optimCV_verbose = optimCV_verbose



        self.no_exps = len(set_of_ss) # the set of sample sizes
        self.no_tasks = 2   # this is fixed as we only have 2 functions for the Borehole example



        self.large_saved_MC_ests_tensor       = torch.zeros( self.no_exps, self.no_replica, self.no_tasks)

        self.large_save_est_scalar_f1_tensor  = torch.zeros(self.no_exps, self.no_replica, self.no_epochs)
        self.large_save_est_scalar_f2_tensor  =  torch.zeros(self.no_exps, self.no_replica, self.no_epochs)

        self.large_save_est_vecfunc_tensor    = torch.zeros(self.no_exps, self.no_replica, self.no_epochs, self.no_tasks)
        self.large_save_closed_form_sols_tensor = torch.zeros(self.no_exps, self.no_replica, self.no_tasks)

        self.large_save_est_vecfunc_fixB_tensor = torch.zeros(self.no_exps, self.no_replica, self.no_epochs, self.no_tasks)
        self.large_save_closed_form_sols_fixB_tensor = torch.zeros(self.no_exps, self.no_replica, self.no_tasks)


    def run_borehole(self, if_plt=True):
        for i in range(self.no_exps):
            cur_ss = self.set_of_ss[i]
            cur_bs_tune = self.set_of_batch_size_tune[i]
            cur_lr_tune = self.set_of_lr_tune[i]
            cur_epochs_tune = self.set_of_tune_epochs[i]
            cur_regularizer_const_weights_optimCV= self.set_of_regularizer_const_weights_optimCV[i]
            cur_bs_optimCV = self.set_of_batch_size_optimCV[i]
            cur_lr_optimCV = self.set_of_lr_optimCV[i]


            no_replica, no_epochs, \
                 large_saved_MC_ests, \
                 large_save_est_scalar_f1, large_save_est_scalar_f2, \
                 large_save_est_vecfunc_fixB, large_save_closed_form_sols_fixB, \
                 large_save_est_vecfunc, large_save_closed_form_sols = EXP_borehole_balanced(no_replica=self.no_replica, no_epochs=self.no_epochs,\
                                                                                              no_points_per_func=cur_ss,\
                                                                                              my_batch_size_tune=cur_bs_tune,\
                                                                                              my_lr_tune=cur_lr_tune, my_tune_epochs=cur_epochs_tune,\
                                                                                              my_tune_verbose=self.tune_verbose,\
                                                                                              my_regularizer_const_weights_optimCV=cur_regularizer_const_weights_optimCV,\
                                                                                              my_batch_size_optimCV=cur_bs_optimCV,\
                                                                                              my_lr_optimCV=cur_lr_optimCV)




            self.large_saved_MC_ests_tensor[i,] = large_saved_MC_ests

            self.large_save_est_scalar_f1_tensor[i,] = large_save_est_scalar_f1
            self.large_save_est_scalar_f2_tensor[i,] = large_save_est_scalar_f2

            self.large_save_est_vecfunc_tensor[i,] = large_save_est_vecfunc
            self.large_save_closed_form_sols_tensor[i,] = large_save_closed_form_sols

            self.large_save_est_vecfunc_fixB_tensor[i,] = large_save_est_vecfunc_fixB
            self.large_save_closed_form_sols_fixB_tensor[i,] = large_save_closed_form_sols_fixB


            if if_plt==True:
                tv_1 = 57.9472
                tv_2 = 72.8904
                true_vals = [tv_1, tv_2]

                import seaborn as sns
                true_vals = [tv_1, tv_2]
                fig, ax = plt.subplots()
                sns.set_style("darkgrid")
                clrs = sns.color_palette("Paired")
                start_pos = 0
                plt.xlabel('Number of Epochs')
                plt.ylabel('Abs. Err.')


                mc_f1_mean = (large_saved_MC_ests[:, 0] - true_vals[0]).abs().mean().repeat(1, no_epochs)
                mc_f2_mean = (large_saved_MC_ests[:, 1] - true_vals[1]).abs().mean().repeat(1, no_epochs)
                mc_f1_std = (large_saved_MC_ests[:, 0] - true_vals[0]).abs().std(dim=0) / (torch.ones(1) * no_replica).sqrt().repeat(1, no_epochs)
                mc_f2_std = (large_saved_MC_ests[:, 1] - true_vals[1]).abs().std(dim=0) / (torch.ones(1) * no_replica).sqrt().repeat(1, no_epochs)

                plt.hlines(mc_f2_mean, start_pos + 1, no_epochs, colors='g',label='MC-HF')  # HF for High-Fidelity function

                plt.hlines((large_save_closed_form_sols[:, 1] - true_vals[1]).abs().mean().detach().numpy(),start_pos + 1, no_epochs, colors='g', linestyles='dotted', label='CF-HF')

                sv_f1_mean = (large_save_est_scalar_f1 - true_vals[0]).abs().mean(dim=0).detach().numpy()
                sv_f2_mean = (large_save_est_scalar_f2 - true_vals[1]).abs().mean(dim=0).detach().numpy()
                sv_f1_std = (large_save_est_scalar_f1 - true_vals[0]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica)
                sv_f2_std = (large_save_est_scalar_f2 - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt( no_replica)


                vv_f1_mean_fixB = (large_save_est_vecfunc_fixB[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
                vv_f2_mean_fixB = (large_save_est_vecfunc_fixB[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
                vv_f1_std_fixB = (large_save_est_vecfunc_fixB[:, :, 0] - true_vals[0]).abs().std( dim=0).detach().numpy() / np.sqrt(no_replica)
                vv_f2_std_fixB = (large_save_est_vecfunc_fixB[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica)
                ax.plot(np.array(list(range(no_epochs))[start_pos:]) + 1, (large_save_est_vecfunc_fixB[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[ start_pos:], c=clrs[7], marker='x', label='vv-CV-FixB-HF')
                ax.fill_between(np.array(list(range(no_epochs))[start_pos:]) + 1, vv_f2_mean_fixB[start_pos:] - vv_f2_std_fixB[start_pos:], vv_f2_mean_fixB[start_pos:] + vv_f2_std_fixB[start_pos:], alpha=0.3, facecolor=clrs[7])

                vv_f1_mean = (large_save_est_vecfunc[:, :, 0] - true_vals[0]).abs().mean(dim=0).detach().numpy()
                vv_f2_mean = (large_save_est_vecfunc[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()
                vv_f1_std = (large_save_est_vecfunc[:, :, 0] - true_vals[0]).abs().std( dim=0).detach().numpy() / np.sqrt(no_replica)
                vv_f2_std = (large_save_est_vecfunc[:, :, 1] - true_vals[1]).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica)
                ax.plot(np.array(list(range(no_epochs))[start_pos:]) + 1, (large_save_est_vecfunc[:, :, 1] - true_vals[1]).abs().mean(dim=0).detach().numpy()[start_pos:],  c=clrs[9], marker='.', label='vv-CV-HF')
                ax.fill_between(np.array(list(range(no_epochs))[start_pos:]) + 1, vv_f2_mean[start_pos:] - vv_f2_std[start_pos:],  vv_f2_mean[start_pos:] + vv_f2_std[start_pos:], alpha=0.3, facecolor=clrs[9])


                ax.legend()
                plt.show()







# Run~
The_borehole_balanced_exps = Borehole_balanced_exps(set_of_ss=[10, 20, 30, 40 , 50, 100, 150], no_replica=100, no_epochs=600, \
                                                    set_of_batch_size_tune=np.repeat(5,7), set_of_lr_tune=np.repeat(0.05,7), set_of_tune_epochs=np.repeat(20,7), tune_verbose=True, \
                                                    set_of_regularizer_const_weights_optimCV=np.repeat(1e-5, 7), set_of_batch_size_optimCV=np.repeat(10,7), \
                                                    set_of_lr_optimCV=[0.09, 0.06, 0.04, 0.02, 0.012, 0.0035, 0.002], optimCV_verbose=True)

no_replica = 100
The_borehole_balanced_exps.run_borehole(if_plt=True)


# no.exps * no.replica * no.epochs * no.tasks










# vv-CV-MD1P: Mean Abs Err
tv_1 = 57.9472  # when mu_r=100 and  sample size is 500000
tv_2 = 72.8904  #
true_vals = [tv_1, tv_2]

(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[0,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=10
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[1,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=20
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[2,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=30
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[3,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=40
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[4,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=50
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[5,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=100
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[6,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=150

# vv-CV-MD1P: Std of Mean Abs Err
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[0, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=10
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[1, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=20
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[2, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=30
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[3, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=40
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[4, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=50
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[5, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=100
(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[6, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=150







 # vv-CV-FixbB-MD1P: Mean Abs Err
tv_1 = 57.9472  # when mu_r=100 and  sample size is 500000
tv_2 = 72.8904  #
true_vals = [tv_1, tv_2]

(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[0,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=10
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[1,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=20
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[2,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=30
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[3,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=40
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[4,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=50
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[5,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=100
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[6,:,-1,1] - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of pi[f2] when ss=150

# vv-CV-fixB-MD1P: Std of Mean Abs Err
no_replica = The_borehole_balanced_exps.no_replica
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[0, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=10
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[1, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=20
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[2, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=30
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[3, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=40
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[4, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=50
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[5, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=100
(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[6, :, -1, 1] - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica)  ## std of mean abs err of pi[f2] when ss=150



# CF: Mean Abs Err
The_borehole_balanced_exps.large_save_closed_form_sols_tensor.size()    # torch.Size([7, 20, 2])  no_ss * no_replica * no_tasks
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[0,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of CF est for Pi[f2] when ss=10
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[1,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of CF est for Pi[f2] when ss=20
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[2,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of CF est for Pi[f2] when ss=30
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[3,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of CF est for Pi[f2] when ss=40
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[4,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of CF est for Pi[f2] when ss=50
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[5,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of CF est for Pi[f2] when ss=100
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[6,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of CF est for Pi[f2] when ss=150



# CF: Std of Mean Abs Err
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[0,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of CF est for Pi[f2] when ss=10
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[1,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of CF est for Pi[f2] when ss=20
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[2,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of CF est for Pi[f2] when ss=30
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[3,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of CF est for Pi[f2] when ss=40
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[4,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of CF est for Pi[f2] when ss=50
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[5,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of CF est for Pi[f2] when ss=100
(The_borehole_balanced_exps.large_save_closed_form_sols_tensor[6,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of CF est for Pi[f2] when ss=150




# MC: Mean Abs Err
The_borehole_balanced_exps.large_saved_MC_ests_tensor.size()  # torch.Size([7, 20, 2])  no_ss * no_replica * no_tasks
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[0,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of MC est for Pi[f2] when ss=10
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[1,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of MC est for Pi[f2] when ss=20
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[2,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of MC est for Pi[f2] when ss=30
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[3,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of MC est for Pi[f2] when ss=40
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[4,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of MC est for Pi[f2] when ss=50
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[5,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of MC est for Pi[f2] when ss=100
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[6,:,1]  - true_vals[1]).abs().mean().detach().numpy() ## mean abs err of MC est for Pi[f2] when ss=150

# MC: Std of Mean Abs err
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[0,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of MC est for Pi[f2] when ss=10
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[1,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of MC est for Pi[f2] when ss=20
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[2,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of MC est for Pi[f2] when ss=30
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[3,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of MC est for Pi[f2] when ss=40
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[4,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of MC est for Pi[f2] when ss=50
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[5,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of MC est for Pi[f2] when ss=100
(The_borehole_balanced_exps.large_saved_MC_ests_tensor[6,:,1]  - true_vals[1]).abs().std().detach().numpy() / np.sqrt(no_replica) ## std of mean abs err of MC est for Pi[f2] when ss=150





# Plot
# Form a pd.dataframe
no_replica = The_borehole_balanced_exps.no_replica
set_of_ss = The_borehole_balanced_exps.set_of_ss
no_ss = len(set_of_ss)


for i in range(no_ss):
    # vv-CV with fixed B
    VV_fixB_cvest_funcidx_methodidx_f1 = list(zip(np.abs(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[i, :, -1, 0].detach().numpy() - true_vals[0]), np.repeat('LF', no_replica), np.repeat('vv-CV-FixB', no_replica), np.repeat("SS={}".format(set_of_ss[i]), no_replica)))
    cur_vv_CV_fixB_est_f1_df = pd.DataFrame(data=VV_fixB_cvest_funcidx_methodidx_f1, columns=['cv_est', 'func_idx', 'method_idx', 'sample_size'])
    if i == 0:
        vv_CV_fixB_est_f1_df = cur_vv_CV_fixB_est_f1_df
    if i >= 1:
        vv_CV_fixB_est_f1_df = vv_CV_fixB_est_f1_df.append(cur_vv_CV_fixB_est_f1_df)

    VV_fixB_cvest_funcidx_methodidx_f2 = list(zip(np.abs(The_borehole_balanced_exps.large_save_est_vecfunc_fixB_tensor[i, :, -1, 1].detach().numpy() - true_vals[1]), np.repeat('HF', no_replica), np.repeat('vv-CV-FixB', no_replica), np.repeat("SS={}".format(set_of_ss[i]), no_replica)))
    cur_vv_CV_fixB_est_f2_df = pd.DataFrame(data=VV_fixB_cvest_funcidx_methodidx_f2, columns=['cv_est', 'func_idx', 'method_idx', 'sample_size'])
    if i == 0:
        vv_CV_fixB_est_f2_df = cur_vv_CV_fixB_est_f2_df
    if i >= 1:
        vv_CV_fixB_est_f2_df = vv_CV_fixB_est_f2_df.append(cur_vv_CV_fixB_est_f2_df)

    # vv-CV with learning B
    VV_cvest_funcidx_methodidx_f1 = list(zip(np.abs(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[i, :, -1,0].detach().numpy() - true_vals[0]), np.repeat('LF',no_replica), np.repeat('vv-CV', no_replica), np.repeat("SS={}".format(set_of_ss[i]), no_replica)))
    cur_vv_CV_est_f1_df = pd.DataFrame(data= VV_cvest_funcidx_methodidx_f1, columns=['cv_est', 'func_idx', 'method_idx', 'sample_size'])
    if i == 0:
        vv_CV_est_f1_df = cur_vv_CV_est_f1_df
    if i >= 1:
        vv_CV_est_f1_df= vv_CV_est_f1_df.append(cur_vv_CV_est_f1_df)

    VV_cvest_funcidx_methodidx_f2 = list(zip(np.abs(The_borehole_balanced_exps.large_save_est_vecfunc_tensor[i, :, -1, 1].detach().numpy() - true_vals[1]), np.repeat('HF', no_replica),np.repeat('vv-CV', no_replica), np.repeat("SS={}".format(set_of_ss[i]), no_replica)))
    cur_vv_CV_est_f2_df = pd.DataFrame(data=VV_cvest_funcidx_methodidx_f2,columns=['cv_est', 'func_idx', 'method_idx', 'sample_size'])
    if i == 0:
        vv_CV_est_f2_df = cur_vv_CV_est_f2_df
    if i >= 1:
        vv_CV_est_f2_df= vv_CV_est_f2_df.append(cur_vv_CV_est_f2_df)





# Merge into one giant dataset
my_vv_CV_DF = vv_CV_est_f1_df.append([vv_CV_est_f2_df, vv_CV_fixB_est_f1_df, vv_CV_fixB_est_f2_df])
my_vv_CV_DF['cv_est']


##################
# Save output
##################
# If you want to rerun the above experiment and save your own results, please uncomment the following line to save your data.

my_vv_CV_DF.to_pickle("../data/Balanced_Borehole_Example_PDframe.pkl")


 