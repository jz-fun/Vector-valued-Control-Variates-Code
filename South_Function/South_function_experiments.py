import pickle
from src.src_vvCV_MDMP.vv_CV_MDMP import *

from South_Function.South_function_trainer import *
##
# Example for vv_CV_MDMP
def my_func_1(X):
    return 1 + X+ X**2 + torch.sin(X * math.pi) * torch.exp(-1.* X.pow(2))



def my_func_2(X):
    return 1.5 + X+ 1.5*(X**2) + 1.75*torch.sin(X * math.pi) * torch.exp(-1.* X.pow(2))





## Varying one of distributions -- check the effect of the closeness of target distributions
mu_1_sit0 = torch.zeros(1,1)
cov_1_sit0= torch.eye(1)
mu_2_sit0 = torch.zeros(1,1)
cov_2_sit0= torch.eye(1)
means_tuple_sit0 = (mu_1_sit0, mu_2_sit0)
covs_tuple_sit0 = (cov_1_sit0, cov_2_sit0)


mu_1_sit1 = torch.zeros(1,1)
cov_1_sit1= torch.eye(1)
mu_2_sit1 = torch.zeros(1,1)
cov_2_sit1= torch.eye(1) * 1.1
means_tuple_sit1 = (mu_1_sit1, mu_2_sit1)
covs_tuple_sit1 = (cov_1_sit1, cov_2_sit1)

mu_1_sit2 = torch.zeros(1,1)
cov_1_sit2= torch.eye(1)
mu_2_sit2 = torch.zeros(1,1)
cov_2_sit2= torch.eye(1) * 1.15
means_tuple_sit2 = (mu_1_sit2, mu_2_sit2)
covs_tuple_sit2 = (cov_1_sit2, cov_2_sit2)

mu_1_sit3 = torch.zeros(1,1)
cov_1_sit3= torch.eye(1)
mu_2_sit3 = torch.zeros(1,1)
cov_2_sit3= torch.eye(1) * 1.2
means_tuple_sit3 = (mu_1_sit3, mu_2_sit3)
covs_tuple_sit3 = (cov_1_sit3, cov_2_sit3)


mu_1_sit4 = torch.zeros(1,1)
cov_1_sit4= torch.eye(1)
mu_2_sit4 = torch.zeros(1,1)
cov_2_sit4= torch.eye(1) * 1.25
means_tuple_sit4 = (mu_1_sit4, mu_2_sit4)
covs_tuple_sit4 = (cov_1_sit4, cov_2_sit4)

tuple_of_meanscovstuple = ((means_tuple_sit0, covs_tuple_sit0), (means_tuple_sit1, covs_tuple_sit1),(means_tuple_sit2, covs_tuple_sit2), (means_tuple_sit3, covs_tuple_sit3), (means_tuple_sit4, covs_tuple_sit4))


#
true_vals = torch.Tensor([[2, 3],[2, 3.15], [2, 3.225], [2, 3.3], [2, 3.375]])
true_vals.size() # 2
true_vals[0].size()


# Initialize the class
no_replica = 100
set_of_ss = 50
no_sets = 5


my_example = toy_example_MDMP(funcs= (my_func_1,  my_func_2), sample_size_per_dist = set_of_ss, num_rep = no_replica, \
                               vv_CV_model=VV_CV_vectorvaluedfuncs_model_MDMP, \
                               vv_CV_obj = penalized_ls_objective_vectorvaluedfunc_MDMP, \
                               prior_kernel = stein_matrix_valued_kernel , base_kernel=rbf_kernel, \
                               batch_size_tune = 5, flag_if_use_medianheuristic=False, beta_cstkernel=0, lr_tune=0.05,\
                               epochs_tune=30, verbose_tune=False, \
                               regularizer_const = 1e-3, regularizer_const_FB=1, batch_size=5, lr=1e-3, epochs=400, \
                               verbose=False)



# Run the algorithm and save outputs
MyvvCV_ests, MysvCV_ests, MysvCV_closed_form_sols = my_example.varying_distrbutions_multiruns(tuple_of_meanscovstuple)

#
MSE_MyvvCV_ests = torch.zeros(len(tuple_of_meanscovstuple))
MSE_MysvCV_ests = torch.zeros(len(tuple_of_meanscovstuple))
MSE_MysvCV_closed_form_sols = torch.zeros(len(tuple_of_meanscovstuple))

#
MSE_MyvvCV_ests_std = torch.zeros(len(tuple_of_meanscovstuple))
MSE_MysvCV_ests_std = torch.zeros(len(tuple_of_meanscovstuple))
MSE_MysvCV_closed_form_sols_std = torch.zeros(len(tuple_of_meanscovstuple))

#
for i in range(len(tuple_of_meanscovstuple)):
    cur_task_true_vals = true_vals[i].unsqueeze(dim=0)
    assert cur_task_true_vals.size() == torch.Size([1, len(tuple_of_meanscovstuple[0][0])])

    MSE_MyvvCV_ests[i] = (MyvvCV_ests[i,:,:] - cur_task_true_vals).pow(2).mean()
    MSE_MyvvCV_ests_std[i] = (MyvvCV_ests[i,:,:] - cur_task_true_vals).pow(2).std()/((len(tuple_of_meanscovstuple) * torch.ones(1)).sqrt())

    MSE_MysvCV_ests[i] =  (MysvCV_ests[i,:,:] - cur_task_true_vals).pow(2).mean()
    MSE_MysvCV_ests_std[i] = (MysvCV_ests[i,:,:] - cur_task_true_vals).pow(2).std()/((len(tuple_of_meanscovstuple) * torch.ones(1)).sqrt())

    MSE_MysvCV_closed_form_sols[i] = (MysvCV_closed_form_sols[i,:,:] - cur_task_true_vals).pow(2).mean()
    MSE_MysvCV_closed_form_sols_std[i] = (MysvCV_closed_form_sols[i,:,:] - cur_task_true_vals).pow(2).std()/((len(tuple_of_meanscovstuple) * torch.ones(1)).sqrt())


MSE_dat = torch.stack((MSE_MyvvCV_ests, MSE_MysvCV_ests, MSE_MysvCV_closed_form_sols), dim=0).detach().numpy()
MSE_dat


# Plot
import numpy as np
import pandas as pd


# Form a pd.dataframe
for i in range(no_sets):
    # vv-CV
    VV_cvest_funcidx_methodidx_f1 = list(zip(np.abs(MyvvCV_ests[i, :, 0].detach().numpy() - true_vals[i, 0].detach().numpy())**2, np.repeat('vv-CV', no_replica), np.repeat("Set {}".format(i), no_replica)))
    cur_vv_CV_est_f1_df = pd.DataFrame(data=VV_cvest_funcidx_methodidx_f1, columns=['cv_est', 'method_idx', 'setting'])
    if i == 0:
        vv_CV_est_f1_df = cur_vv_CV_est_f1_df
    if i >= 1:
        vv_CV_est_f1_df = vv_CV_est_f1_df.append(cur_vv_CV_est_f1_df)

    VV_cvest_funcidx_methodidx_f2 = list(zip(np.abs(MyvvCV_ests[i, :, 1].detach().numpy() - true_vals[i, 1].detach().numpy())**2, np.repeat('vv-CV', no_replica), np.repeat("Set {}".format(i), no_replica)))
    cur_vv_CV_est_f2_df = pd.DataFrame(data=VV_cvest_funcidx_methodidx_f2, columns=['cv_est', 'method_idx', 'setting'])
    if i == 0:
        vv_CV_est_f2_df = cur_vv_CV_est_f2_df
    if i >= 1:
        vv_CV_est_f2_df = vv_CV_est_f2_df.append(cur_vv_CV_est_f2_df)

    vv_CV_est_giant_f1f2 = vv_CV_est_f1_df.append(vv_CV_est_f2_df)


    # CF -- should use sv-CV_closed form sols
    CF_cvest_funcidx_methodidx_f1 = list(zip(np.abs(MysvCV_closed_form_sols[i, :, 0].detach().numpy() - true_vals[i, 0].detach().numpy())**2, np.repeat('CF', no_replica), np.repeat("Set {}".format(i), no_replica)))
    cur_CF_est_f1_df = pd.DataFrame(data=CF_cvest_funcidx_methodidx_f1, columns=['cv_est', 'method_idx', 'setting'])
    if i == 0:
        CF_est_f1_df = cur_CF_est_f1_df
    if i >= 1:
        CF_est_f1_df = CF_est_f1_df.append(cur_CF_est_f1_df)

    CF_cvest_funcidx_methodidx_f2 = list(zip(np.abs(MysvCV_closed_form_sols[i, :, 1].detach().numpy() - true_vals[i, 1].detach().numpy())**2, np.repeat('CF', no_replica), np.repeat("Set {}".format(i), no_replica)))
    cur_CF_est_f2_df = pd.DataFrame(data=CF_cvest_funcidx_methodidx_f2, columns=['cv_est', 'method_idx', 'setting'])
    if i == 0:
        CF_est_f2_df = cur_CF_est_f2_df
    if i >= 1:
        CF_est_f2_df = CF_est_f2_df.append(cur_CF_est_f2_df)

    CF_est_giant_f1f2 = CF_est_f1_df.append(CF_est_f2_df)



    # sv-CV
    SV_cvest_funcidx_methodidx_f1 = list(zip(np.abs(MysvCV_ests[i, :, 0].detach().numpy() - true_vals[i, 0].detach().numpy())**2, np.repeat('CV', no_replica), np.repeat("Set {}".format(i), no_replica)))
    cur_sv_CV_est_f1_df = pd.DataFrame(data=SV_cvest_funcidx_methodidx_f1, columns=['cv_est', 'method_idx', 'setting'])
    if i == 0:
        sv_CV_est_f1_df = cur_sv_CV_est_f1_df
    if i >= 1:
        sv_CV_est_f1_df = sv_CV_est_f1_df.append(cur_sv_CV_est_f1_df)

    SV_cvest_funcidx_methodidx_f2 = list(zip(np.abs(MysvCV_ests[i, :, 1].detach().numpy()- true_vals[i, 1].detach().numpy())**2, np.repeat('CV', no_replica), np.repeat("Set {}".format(i), no_replica)))
    cur_sv_CV_est_f2_df = pd.DataFrame(data=SV_cvest_funcidx_methodidx_f2, columns=['cv_est', 'method_idx', 'setting'])
    if i == 0:
        sv_CV_est_f2_df = cur_sv_CV_est_f2_df
    if i >= 1:
        sv_CV_est_f2_df = sv_CV_est_f2_df.append(cur_sv_CV_est_f2_df)

    sv_CV_est_giant_f1f2 = sv_CV_est_f1_df.append(sv_CV_est_f2_df)


# Merge into one giant dataset
my_vv_CV_DF_giant_f1f2 = vv_CV_est_giant_f1f2.append([CF_est_giant_f1f2, sv_CV_est_giant_f1f2])
my_vv_CV_DF_giant_f1f2['cv_est']









##################
# Save output
##################
# Comments: If you want to rerun the above experiment and save your own results, please uncomment the following block to save your data.
#
# my_vv_CV_DF_giant_f1f2 = vv_CV_est_giant_f1f2.append([sv_CV_est_giant_f1f2])
#
# my_vv_CV_DF_giant_f1f2.to_pickle("South_function_pdframe_saved.pkl")
#
# with open('South_function_all_data.pkl', 'wb') as output:
#     MSE_MyvvCV_ests_std = MSE_MyvvCV_ests_std
#     pickle.dump(MSE_MyvvCV_ests_std, output, pickle.HIGHEST_PROTOCOL)
#
#     MSE_MysvCV_ests_std = MSE_MysvCV_ests_std
#     pickle.dump(MSE_MysvCV_ests_std, output, pickle.HIGHEST_PROTOCOL)
#
#     MSE_MysvCV_closed_form_sols_std = MSE_MysvCV_closed_form_sols_std
#     pickle.dump(MSE_MysvCV_closed_form_sols_std, output, pickle.HIGHEST_PROTOCOL)
#
#     #
#     MyvvCV_ests = MyvvCV_ests
#     pickle.dump(MyvvCV_ests, output, pickle.HIGHEST_PROTOCOL)
#
#     MysvCV_ests = MysvCV_ests
#     pickle.dump(MysvCV_ests, output, pickle.HIGHEST_PROTOCOL)
#
#     MysvCV_closed_form_sols = MysvCV_closed_form_sols
#     pickle.dump(MysvCV_closed_form_sols, output, pickle.HIGHEST_PROTOCOL)
#
#     MSE_dat = MSE_dat
#     pickle.dump(MSE_dat, output, pickle.HIGHEST_PROTOCOL)
#
#     means_tuple_sit1 = means_tuple_sit1
#     pickle.dump(means_tuple_sit1, output, pickle.HIGHEST_PROTOCOL)
#     covs_tuple_sit1 = covs_tuple_sit1
#     pickle.dump(covs_tuple_sit1, output, pickle.HIGHEST_PROTOCOL)
#
#     means_tuple_sit2 = means_tuple_sit2
#     pickle.dump(means_tuple_sit2, output, pickle.HIGHEST_PROTOCOL)
#     covs_tuple_sit2 = covs_tuple_sit2
#     pickle.dump(covs_tuple_sit2, output, pickle.HIGHEST_PROTOCOL)
#
#     means_tuple_sit3 = means_tuple_sit3
#     pickle.dump(means_tuple_sit3, output, pickle.HIGHEST_PROTOCOL)
#     covs_tuple_sit3 = covs_tuple_sit3
#     pickle.dump(covs_tuple_sit3, output, pickle.HIGHEST_PROTOCOL)
#
#     means_tuple_sit4 = means_tuple_sit4
#     pickle.dump(means_tuple_sit4, output, pickle.HIGHEST_PROTOCOL)
#     covs_tuple_sit4 = covs_tuple_sit4
#     pickle.dump(covs_tuple_sit4, output, pickle.HIGHEST_PROTOCOL)
#
