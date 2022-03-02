import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Borehole_Example.src_ProdKernel_vvCV_MD1P.vv_CV_unbalanced_FixB_MD1P_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.vv_CV_unbalanced_MD1P_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.stein_operators_borehole import *
from Borehole_Example.src_ProdKernel_vvCV_MD1P.product_base_kernels_borehole import *





def my_func_LF(X):
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


def my_func_HF(X):
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










def EXP_borehole_unbalanced(no_replica=20, no_epochs=800, no_points_HF=50, no_points_LF=100, my_batch_size_tune=5, my_lr_tune=0.2, my_tune_epochs=10, my_tune_verbose=True, my_regularizer_const_weights_optimCV=1e-5, my_batch_size_optimCV = 10, my_lr_optimCV=0.004, my_optimCV_verbose=True):
    NO_tasks= 2

    large_saved_MC_ests = torch.zeros(no_replica, NO_tasks)
    large_save_est_scalar_f1 = torch.zeros(no_replica, no_epochs)
    large_save_est_scalar_f2 = torch.zeros(no_replica, no_epochs)
    large_save_est_vecfunc = torch.zeros(no_replica, no_epochs, NO_tasks)
    large_save_est_vecfunc_fixB = torch.zeros(no_replica, no_epochs, NO_tasks)


    for i in range(no_replica):
        # Setting means and vars
        mu_r_w = torch.ones(1) * 0.1
        mu_r = torch.ones(1) * 100.
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
        my_mus = my_mus.unsqueeze(dim=1)
        my_mus.size()

        my_vars = torch.Tensor([var_r_w, var_r, var_T_u, var_T_l, var_H_u, var_H_l, var_L, var_K_w])
        my_vars = my_vars.unsqueeze(dim=1)
        my_vars.size()


        print("REP {} out of {}-----------".format(i + 1, no_replica))
        # Training samples
        m_HF = no_points_HF
        m_LF = no_points_LF

        torch.manual_seed(2 * i )
        r_ws_X1 = mu_r_w + torch.sqrt(var_r_w) * torch.randn(m_HF, 1)
        rs_X1 = mu_r + torch.sqrt(var_r) * torch.randn(m_HF, 1)
        T_us_X1 = mu_T_u + torch.sqrt(var_T_u) * torch.randn(m_HF, 1)
        T_ls_X1 = mu_T_l + torch.sqrt(var_T_l) * torch.randn(m_HF, 1)
        H_us_X1 = mu_H_u + torch.sqrt(var_H_u) * torch.randn(m_HF, 1)
        H_ls_X1 = mu_H_l + torch.sqrt(var_H_l) * torch.randn(m_HF, 1)
        Ls_X1 = mu_L + torch.sqrt(var_L) * torch.randn(m_HF, 1)
        K_ws_X1 = mu_K_w + torch.sqrt(var_K_w) * torch.randn(m_HF, 1)

        X1 = torch.stack((r_ws_X1, rs_X1, T_us_X1, T_ls_X1, H_us_X1, H_ls_X1, Ls_X1, K_ws_X1), dim=1).squeeze()
        X1.size()
        Y1 = my_func_LF(X1)
        Y1.size()


        torch.manual_seed(2 * i+1)
        r_ws_X2 = mu_r_w + torch.sqrt(var_r_w) * torch.randn(m_LF, 1)
        rs_X2 = mu_r + torch.sqrt(var_r) * torch.randn(m_LF, 1)
        T_us_X2 = mu_T_u + torch.sqrt(var_T_u) * torch.randn(m_LF, 1)
        T_ls_X2 = mu_T_l + torch.sqrt(var_T_l) * torch.randn(m_LF, 1)
        H_us_X2 = mu_H_u + torch.sqrt(var_H_u) * torch.randn(m_LF, 1)
        H_ls_X2 = mu_H_l + torch.sqrt(var_H_l) * torch.randn(m_LF, 1)
        Ls_X2 = mu_L + torch.sqrt(var_L) * torch.randn(m_LF, 1)
        K_ws_X2 = mu_K_w + torch.sqrt(var_K_w) * torch.randn(m_LF, 1)

        X2 = torch.stack((r_ws_X2, rs_X2, T_us_X2, T_ls_X2, H_us_X2, H_ls_X2, Ls_X2, K_ws_X2), dim=1).squeeze()
        X2.size()
        Y2 = my_func_HF(X2)
        Y2.size()

        # Compute scores
        score_X1 = product_Normal_score(my_mus, my_vars, X1)
        score_X1.size()
        score_X2 = product_Normal_score(my_mus, my_vars, X2)
        score_X2.size()

        xall = (X1, X2)
        yall = (Y1, Y2)
        score_all = (score_X1, score_X2)

        # Monte Carlo estimates
        large_saved_MC_ests[i] = torch.Tensor([Y1.mean(dim=0), Y2.mean(dim=0)])


        # vv-CV-unbalanced: MD1P with B fixed
        print("REP {} out of {} --- vv-CV-unbalanced: MD1P with B fixed -----------".format(i + 1, no_replica))
        my_SCV_vectorvaluedfunc_unbalanced_fixB = VV_CV_vectorvaluedfuncs_model_unbalanced_fixB_borehole(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_unbalanced_fixB_borehole, prior_kernel=stein_base_kernel_borehole, base_kernel=prod_rbf_kernel_Borehore, Xs_tuple=xall, Ys_tuple=yall, scores_tuple=score_all)
        torch.manual_seed(0)
        my_SCV_vectorvaluedfunc_unbalanced_fixB.do_tune_kernelparams_negmllk(batch_size_tune=my_batch_size_tune, flag_if_use_medianheuristic=False, beta_cstkernel=1, lr=my_lr_tune, epochs=my_tune_epochs, verbose=my_tune_verbose)
        #  Mannualy set a B
        my_SCV_vectorvaluedfunc_unbalanced_fixB.B = 0.005 * torch.Tensor([[0.1, 0.01], [0.01, 0.1]])
        my_SCV_vectorvaluedfunc_unbalanced_fixB.do_optimize_vv_CV(regularizer_const=my_regularizer_const_weights_optimCV, batch_size=np.int(my_batch_size_optimCV/NO_tasks), lr=my_lr_optimCV, epochs=no_epochs, verbose=my_optimCV_verbose)
        large_save_est_vecfunc_fixB[i] = my_SCV_vectorvaluedfunc_unbalanced_fixB.saved_BQ_est.squeeze().detach().clone()




        # vv-CV-unbalanced: MD1P with learning B
        print("REP {} out of {} --- vv-CV-unbalanced: MD1P with learning B -----------".format(i + 1, no_replica))
        torch.manual_seed(0)
        my_SCV_vectorvaluedfunc = VV_CV_vectorvaluedfuncs_model_unbalanced_borehole(vv_cv_objective=penalized_ls_objective_vectorvaluedfunc_unbalanced_borehole, prior_kernel=stein_base_kernel_borehole,base_kernel=prod_rbf_kernel_Borehore, Xs_tuple=xall, Ys_tuple=yall, scores_tuple=score_all)
        my_SCV_vectorvaluedfunc.do_tune_kernelparams_negmllk(batch_size_tune=my_batch_size_tune, flag_if_use_medianheuristic=False, beta_cstkernel=1, lr=my_lr_tune, epochs=my_tune_epochs, verbose=my_tune_verbose)
        torch.manual_seed(0)
        my_SCV_vectorvaluedfunc.do_optimize_vv_CV(regularizer_const=my_regularizer_const_weights_optimCV, regularizer_const_FB=1, batch_size=np.int(my_batch_size_optimCV/NO_tasks), lr=my_lr_optimCV, epochs=no_epochs, verbose=my_optimCV_verbose)
        large_save_est_vecfunc[i] = my_SCV_vectorvaluedfunc.saved_BQ_est.squeeze().detach().clone()



    return  no_replica, no_epochs,  large_saved_MC_ests , large_save_est_scalar_f1, large_save_est_scalar_f2 , large_save_est_vecfunc_fixB  , large_save_est_vecfunc









class Borehole_unbalanced_exps(object):
    def __init__(self, set_of_ss_HF, set_of_ss_LF, no_replica, no_epochs, set_of_batch_size_tune, set_of_lr_tune, set_of_tune_epochs, tune_verbose, set_of_regularizer_const_weights_optimCV, set_of_batch_size_optimCV, set_of_lr_optimCV, optimCV_verbose ):
        """
        :param set_of_ss_HF: list, e.g. [50, 50, 50]
        :param set_of_ss_LF: list, e.g. [50, 100,150]
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

        assert len(set_of_ss_LF) == len(set_of_ss_HF), "set_of_ss_LF and set_of_ss_HF should have equal size."

        self.set_of_ss_HF = set_of_ss_HF
        self.set_of_ss_LF = set_of_ss_LF
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



        self.no_exps = len(set_of_ss_LF) # the set of sample sizes, e.g., we have (50,50) (50, 100) (50 150) three experiments
        self.no_tasks = 2   # this is fixed as we only have 2 functions for the Borehole example


        self.large_saved_MC_ests_tensor       = torch.zeros( self.no_exps, self.no_replica, self.no_tasks)

        self.large_save_est_scalar_f1_tensor  = torch.zeros(self.no_exps, self.no_replica, self.no_epochs)
        self.large_save_est_scalar_f2_tensor  =  torch.zeros(self.no_exps, self.no_replica, self.no_epochs)
        self.large_save_est_vecfunc_tensor    = torch.zeros(self.no_exps, self.no_replica, self.no_epochs, self.no_tasks)
        self.large_save_est_vecfunc_fixB_tensor = torch.zeros(self.no_exps, self.no_replica, self.no_epochs, self.no_tasks)


    def run_borehole(self, if_plt=True):
        for i in range(self.no_exps):
            cur_ss_HF = self.set_of_ss_HF[i]
            cur_ss_LF = self.set_of_ss_LF[i]

            cur_bs_tune = self.set_of_batch_size_tune[i]
            cur_lr_tune = self.set_of_lr_tune[i]
            cur_epochs_tune = self.set_of_tune_epochs[i]
            cur_regularizer_const_weights_optimCV= self.set_of_regularizer_const_weights_optimCV[i]
            cur_bs_optimCV = self.set_of_batch_size_optimCV[i]
            cur_lr_optimCV = self.set_of_lr_optimCV[i]


            no_replica, no_epochs, \
                 large_saved_MC_ests, \
                 large_save_est_scalar_f1, large_save_est_scalar_f2, \
                 large_save_est_vecfunc_fixB, \
                 large_save_est_vecfunc = EXP_borehole_unbalanced(no_replica=self.no_replica, no_epochs=self.no_epochs,\
                                                                                              no_points_HF = cur_ss_HF, \
                                                                                              no_points_LF=cur_ss_LF,\
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
            self.large_save_est_vecfunc_fixB_tensor[i,] = large_save_est_vecfunc_fixB


            if if_plt==True:
                tv_LF = 57.9472  # when mu_r=100 and  sample size is 500000
                tv_HF = 72.8904  #

                fig, ax = plt.subplots()
                sns.set_style("darkgrid")
                clrs = sns.color_palette("Paired")
                start_pos = 0
                plt.xlabel('Number of Epochs')
                plt.ylabel('Abs. Err.')

                plt.hlines((large_saved_MC_ests[:, 1] - tv_HF).abs().mean().repeat(1, no_epochs), start_pos + 1, no_epochs, colors='g',label='MC-HF')

                vv_HF_mean_fixB = (large_save_est_vecfunc_fixB[:, :, 1] - tv_HF).abs().mean(dim=0).detach().numpy()
                vv_HF_std_fixB = (large_save_est_vecfunc_fixB[:, :, 1] - tv_HF).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica)
                ax.plot(np.array(list(range(no_epochs))[start_pos:]) + 1, vv_HF_mean_fixB[start_pos:], c=clrs[7], marker='x', label='vv-CV-FixB-HF')
                ax.fill_between(np.array(list(range(no_epochs))[start_pos:]) + 1, vv_HF_mean_fixB[start_pos:] - vv_HF_std_fixB[start_pos:], vv_HF_mean_fixB[start_pos:] + vv_HF_std_fixB[start_pos:], alpha=0.3, facecolor=clrs[7])

                vv_HF_mean = (large_save_est_vecfunc[:, :, 1] - tv_HF).abs().mean(dim=0).detach().numpy()
                vv_HF_std = (large_save_est_vecfunc[:, :, 1] - tv_HF).abs().std(dim=0).detach().numpy() / np.sqrt(no_replica)
                ax.plot(np.array(list(range(no_epochs))[start_pos:]) + 1, vv_HF_mean[start_pos:],  c=clrs[9], marker='.', label='vv-CV-HF')
                ax.fill_between(np.array(list(range(no_epochs))[start_pos:]) + 1, vv_HF_mean[start_pos:] - vv_HF_std[start_pos:],  vv_HF_mean[start_pos:] + vv_HF_std[start_pos:], alpha=0.3, facecolor=clrs[9])

                ax.legend()
                plt.show()





# Run~
The_borehole_unbalanced_exps = Borehole_unbalanced_exps(set_of_ss_HF=[20, 20, 20], set_of_ss_LF=[20, 40, 60], \
                                                        no_replica=100, no_epochs=600, \
                                                        set_of_batch_size_tune=np.repeat(5,3), set_of_lr_tune=np.repeat(0.05,3),set_of_tune_epochs=np.repeat(20,3), tune_verbose=True,\
                                                        set_of_regularizer_const_weights_optimCV=np.repeat(1e-5, 3), set_of_batch_size_optimCV=np.repeat(10,3), set_of_lr_optimCV=[0.06, 0.04, 0.02], optimCV_verbose=True )

The_borehole_unbalanced_exps.run_borehole(if_plt=True)





# Results
tv_LF = 57.9472  # when mu_r=100 and  sample size is 500000
tv_HF = 72.8904  #


# Plot
# Form a pd.dataframe
no_replica = The_borehole_unbalanced_exps.no_replica
set_of_HF = The_borehole_unbalanced_exps.set_of_ss_HF
set_of_LF = The_borehole_unbalanced_exps.set_of_ss_LF
no_ss = len(set_of_HF)



# Note that now the [:,:,:,0] is for HF ~
for i in range(no_ss):
    # vv-CV with fixed B
    VV_fixB_cvest_funcidx_methodidx_f1 = list(zip(np.abs(The_borehole_unbalanced_exps.large_save_est_vecfunc_fixB_tensor[i, :, -1, 0].detach().numpy() - tv_LF), np.repeat('Low-fidelity model', no_replica), np.repeat('Fixed B', no_replica), np.repeat(r"$m_L={}$".format(set_of_LF[i]), no_replica)))
    cur_vv_CV_fixB_est_f1_df = pd.DataFrame(data=VV_fixB_cvest_funcidx_methodidx_f1, columns=['cv_est', 'func_idx', 'method_idx', 'sample_size'])
    if i == 0:
        vv_CV_fixB_est_f1_df = cur_vv_CV_fixB_est_f1_df
    if i >= 1:
        vv_CV_fixB_est_f1_df = vv_CV_fixB_est_f1_df.append(cur_vv_CV_fixB_est_f1_df)

    VV_fixB_cvest_funcidx_methodidx_f2 = list(zip(np.abs(The_borehole_unbalanced_exps.large_save_est_vecfunc_fixB_tensor[i, :, -1, 1].detach().numpy() - tv_HF), np.repeat('High-fidelity model', no_replica), np.repeat('Fixed B', no_replica), np.repeat(r"$m_L={}$".format(set_of_LF[i]), no_replica)))
    cur_vv_CV_fixB_est_f2_df = pd.DataFrame(data=VV_fixB_cvest_funcidx_methodidx_f2, columns=['cv_est', 'func_idx', 'method_idx', 'sample_size'])
    if i == 0:
        vv_CV_fixB_est_f2_df = cur_vv_CV_fixB_est_f2_df
    if i >= 1:
        vv_CV_fixB_est_f2_df = vv_CV_fixB_est_f2_df.append(cur_vv_CV_fixB_est_f2_df)

    # vv-CV with learning B
    VV_cvest_funcidx_methodidx_f1 = list(zip(np.abs(The_borehole_unbalanced_exps.large_save_est_vecfunc_tensor[i, :, -1,0].detach().numpy() - tv_LF), np.repeat('Low-fidelity model',no_replica), np.repeat('Estimated B', no_replica), np.repeat(r"$m_L={}$".format(set_of_LF[i]), no_replica)))
    cur_vv_CV_est_f1_df = pd.DataFrame(data= VV_cvest_funcidx_methodidx_f1, columns=['cv_est', 'func_idx', 'method_idx', 'sample_size'])
    if i == 0:
        vv_CV_est_f1_df = cur_vv_CV_est_f1_df
    if i >= 1:
        vv_CV_est_f1_df= vv_CV_est_f1_df.append(cur_vv_CV_est_f1_df)

    VV_cvest_funcidx_methodidx_f2 = list(zip(np.abs(The_borehole_unbalanced_exps.large_save_est_vecfunc_tensor[i, :, -1, 1].detach().numpy() - tv_HF), np.repeat('High-fidelity model', no_replica),np.repeat('Estimated B', no_replica), np.repeat(r"$m_L={}$".format(set_of_LF[i]), no_replica)))
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
# Comment: If you want to rerun the above experiment and save your own results, please uncomment the following line to save your data.
# my_vv_CV_DF.to_pickle("Unbalanced_Borehole_Example_PDframe.pkl")





