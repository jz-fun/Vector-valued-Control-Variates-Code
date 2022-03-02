###########
# Plot
###########
#
mu_1_sit4 = torch.zeros(1,1)
cov_1_sit4= torch.eye(1)
mu_2_sit4 = torch.zeros(1,1)
cov_2_sit4= torch.eye(1) * 1.25
means_tuple_sit4 = (mu_1_sit4, mu_2_sit4)
covs_tuple_sit4 = (cov_1_sit4, cov_2_sit4)

#
fig, axs = plt.subplots(3, 3)
fig.set_figwidth(12)
fig.set_figheight(7.5)

#
x_illu = np.linspace(-2, 2, 500)
x_illu_matker = np.linspace(-5, 5, 500)
x_illu_tor = torch.Tensor(x_illu_matker).unsqueeze(dim=1)
x_illu_tor.size()
score_tensor_x_illu_tor = helper_get_scores(x_illu_tor, means_tuple_sit4, covs_tuple_sit4)
projected = torch.ones(1,2,1)


#### Case 1. Base kernel is Squared Exponential Kernel
## For 1^T K_0(, fixpoint)
mat_kernel_illu = stein_matrix_valued_kernel(base_kernel = rbf_kernel, T = 2)
mat_kernel_illu.base_kernel_parm1 = torch.ones(1)
mat_kernel_illu.base_kernel_parm2 = torch.ones(1) * 1
mat_kernel_illu.B = torch.Tensor([[1., 0.1],[0.1, 1.]])

# 1^T K_0(,0)
fix_point = torch.zeros(1,1)
score_tensor_fixpoint = helper_get_scores(fix_point, means_tuple_sit4, covs_tuple_sit4)
Gram_illu = mat_kernel_illu.cal_datasetwise(fix_point, x_illu_tor, score_tensor_fixpoint, score_tensor_x_illu_tor)
Gram_illu = Gram_illu.squeeze()
vv_y_illu = Gram_illu @ projected
vv_y_illu.size()

# 1^T K_0(,1)
fix_point_at1 = torch.ones(1,1)
score_tensor_fixpoint_at1 = helper_get_scores(fix_point_at1, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at1 = mat_kernel_illu.cal_datasetwise(fix_point_at1, x_illu_tor, score_tensor_fixpoint_at1, score_tensor_x_illu_tor)
Gram_illu_at1 = Gram_illu_at1.squeeze()
vv_y_illu_at1 = Gram_illu_at1 @ projected
vv_y_illu_at1.size()

# 1^T K_0(,2)
fix_point_at2 = torch.ones(1,1) * 2
score_tensor_fixpoint_at2 = helper_get_scores(fix_point_at2, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at2 = mat_kernel_illu.cal_datasetwise(fix_point_at2, x_illu_tor, score_tensor_fixpoint_at2, score_tensor_x_illu_tor)
Gram_illu_at2 = Gram_illu_at2.squeeze()
vv_y_illu_at2 = Gram_illu_at2 @ projected
vv_y_illu_at2.size()



#### Case 2. Base kernel is 1-order Polynomial Kernel
## For 1^T K_0(, fixpoint)
mat_kernel_illu_1order = stein_matrix_valued_kernel(base_kernel = polynomial_kernel, T = 2)
mat_kernel_illu_1order.base_kernel_parm1 = torch.ones(1)
mat_kernel_illu_1order.base_kernel_parm2 = torch.ones(1) * 1
mat_kernel_illu_1order.B = torch.Tensor([[1., 0.1],[0.1, 1.]])

# 1^T K_0(,0)
fix_point = torch.zeros(1,1)
score_tensor_fixpoint = helper_get_scores(fix_point, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_1order = mat_kernel_illu_1order.cal_datasetwise(fix_point, x_illu_tor, score_tensor_fixpoint, score_tensor_x_illu_tor)
Gram_illu_1order = Gram_illu_1order.squeeze()
vv_y_illu_1order = Gram_illu_1order @ projected
vv_y_illu_1order.size()

# 1^T K_0(,1)
fix_point_at1 = torch.ones(1,1)
score_tensor_fixpoint_at1 = helper_get_scores(fix_point_at1, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at1_1order = mat_kernel_illu_1order.cal_datasetwise(fix_point_at1, x_illu_tor, score_tensor_fixpoint_at1, score_tensor_x_illu_tor)
Gram_illu_at1_1order = Gram_illu_at1_1order.squeeze()
vv_y_illu_at1_1order = Gram_illu_at1_1order @ projected
vv_y_illu_at1_1order.size()

# 1^T K_0(,2)
fix_point_at2 = torch.ones(1,1) * 2
score_tensor_fixpoint_at2 = helper_get_scores(fix_point_at2, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at2_1order = mat_kernel_illu_1order.cal_datasetwise(fix_point_at2, x_illu_tor, score_tensor_fixpoint_at2, score_tensor_x_illu_tor)
Gram_illu_at2_1order = Gram_illu_at2_1order.squeeze()
vv_y_illu_at2_1order = Gram_illu_at2_1order @ projected
vv_y_illu_at2_1order.size()




#### Case 3. Base kernel is 2-order Polynomial Kernel
## For 1^T K_0(, fixpoint)
mat_kernel_illu_2order = stein_matrix_valued_kernel(base_kernel = polynomial_kernel, T = 2)
mat_kernel_illu_2order.base_kernel_parm1 = torch.ones(1) *2
mat_kernel_illu_2order.base_kernel_parm2 = torch.ones(1) * 1
mat_kernel_illu_2order.B = torch.Tensor([[1., 0.1],[0.1, 1.]])

# 1^T K_0(,0)
fix_point = torch.zeros(1,1)
score_tensor_fixpoint = helper_get_scores(fix_point, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_2order = mat_kernel_illu_2order.cal_datasetwise(fix_point, x_illu_tor, score_tensor_fixpoint, score_tensor_x_illu_tor)
Gram_illu_2order = Gram_illu_2order.squeeze()
vv_y_illu_2order = Gram_illu_2order @ projected
vv_y_illu_2order.size()

# 1^T K_0(,1)
fix_point_at1 = torch.ones(1,1)
score_tensor_fixpoint_at1 = helper_get_scores(fix_point_at1, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at1_2order = mat_kernel_illu_2order.cal_datasetwise(fix_point_at1, x_illu_tor, score_tensor_fixpoint_at1, score_tensor_x_illu_tor)
Gram_illu_at1_2order = Gram_illu_at1_2order.squeeze()
vv_y_illu_at1_2order = Gram_illu_at1_2order @ projected
vv_y_illu_at1_2order.size()

# 1^T K_0(,2)
fix_point_at2 = torch.ones(1,1) * 2
score_tensor_fixpoint_at2 = helper_get_scores(fix_point_at2, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at2_2order = mat_kernel_illu_2order.cal_datasetwise(fix_point_at2, x_illu_tor, score_tensor_fixpoint_at2, score_tensor_x_illu_tor)
Gram_illu_at2_2order = Gram_illu_at2_2order.squeeze()
vv_y_illu_at2_2order = Gram_illu_at2_2order @ projected
vv_y_illu_at2_2order.size()



#### Case 4. Base kernel is 3-order Polynomial Kernel
## For 1^T K_0(, fixpoint)
mat_kernel_illu_3order = stein_matrix_valued_kernel(base_kernel = polynomial_kernel, T = 2)
mat_kernel_illu_3order.base_kernel_parm1 = torch.ones(1) * 3
mat_kernel_illu_3order.base_kernel_parm2 = torch.ones(1) * 1
mat_kernel_illu_3order.B = torch.Tensor([[1., 0.1],[0.1, 1.]])

# 1^T K_0(,0)
fix_point = torch.zeros(1,1)
score_tensor_fixpoint = helper_get_scores(fix_point, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_3order = mat_kernel_illu_3order.cal_datasetwise(fix_point, x_illu_tor, score_tensor_fixpoint, score_tensor_x_illu_tor)
Gram_illu_3order = Gram_illu_3order.squeeze()
vv_y_illu_3order = Gram_illu_3order @ projected
vv_y_illu_3order.size()

# 1^T K_0(,1)
fix_point_at1 = torch.ones(1,1)
score_tensor_fixpoint_at1 = helper_get_scores(fix_point_at1, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at1_3order = mat_kernel_illu_3order.cal_datasetwise(fix_point_at1, x_illu_tor, score_tensor_fixpoint_at1, score_tensor_x_illu_tor)
Gram_illu_at1_3order = Gram_illu_at1_3order.squeeze()
vv_y_illu_at1_3order = Gram_illu_at1_3order @ projected
vv_y_illu_at1_3order.size()

# 1^T K_0(,2)
fix_point_at2 = torch.ones(1,1) * 2
score_tensor_fixpoint_at2 = helper_get_scores(fix_point_at2, means_tuple_sit4, covs_tuple_sit4)
Gram_illu_at2_3order = mat_kernel_illu_3order.cal_datasetwise(fix_point_at2, x_illu_tor, score_tensor_fixpoint_at2, score_tensor_x_illu_tor)
Gram_illu_at2_3order = Gram_illu_at2_3order.squeeze()
vv_y_illu_at2_3order = Gram_illu_at2_3order @ projected
vv_y_illu_at2_3order.size()







# For math symbols: see https://matplotlib.org/2.0.2/users/mathtext.html
axs[0,0].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,0))_1$')
axs[0,0].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,0))_2$')
axs[0,0].set_ylabel('squared expoential k', fontsize=13, rotation='vertical')
axs[0,0].set_xlabel("x", fontsize=15)
axs[0,0].tick_params(labelsize=15)


axs[0,1].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at1[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,1))_1$')
axs[0,1].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at1[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,1))_2$')
axs[0,1].set_ylabel('squared expoential k', fontsize=13, rotation='vertical')
axs[0,1].set_xlabel("x", fontsize=15)
axs[0,1].tick_params(labelsize=15)


axs[0,2].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at2[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,2))_1$') #\cdot
axs[0,2].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at2[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,2))_2$')
axs[0,2].set_ylabel('squared expoential k', fontsize=13, rotation='vertical')
axs[0,2].set_xlabel("x", fontsize=15)
axs[0,2].tick_params(labelsize=15)



#### 1-order Polynomial
axs[1,0].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_1order[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,0))_1$')
axs[1,0].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_1order[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,0))_2$')
axs[1,0].set_ylabel('1st order polyn. k', fontsize=13, rotation='vertical')
axs[1,0].set_xlabel("x", fontsize=15)
axs[1,0].tick_params(labelsize=15)


axs[1,1].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at1_1order[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,1))_1$')
axs[1,1].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at1_1order[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,1))_2$')
axs[1,1].set_ylabel('1st order polyn. k', fontsize=13, rotation='vertical')
axs[1,1].set_xlabel("x", fontsize=15)
axs[1,1].tick_params(labelsize=15)

axs[1,2].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at2_1order[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,2))_1$')
axs[1,2].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at2_1order[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,2))_2$')
axs[1,2].set_ylabel('1st order polyn. k', fontsize=13, rotation='vertical')
axs[1,2].set_xlabel("x", fontsize=15)
axs[1,2].tick_params(labelsize=15)



#### 2-order Polynomial
axs[2,0].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_2order[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,0))_1$')
axs[2,0].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_2order[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,0))_2$')
axs[2,0].set_ylabel('2nd order polyn. k', fontsize=13, rotation='vertical')
axs[2,0].set_xlabel("x", fontsize=15)
axs[2,0].tick_params(labelsize=15)

axs[2,1].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at1_2order[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,1))_1$')
axs[2,1].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at1_2order[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,1))_2$')
axs[2,1].set_ylabel('2nd order polyn. k', fontsize=13, rotation='vertical')
axs[2,1].set_xlabel("x", fontsize=15)
axs[2,1].tick_params(labelsize=15)


axs[2,2].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at2_2order[:,0,:].squeeze().detach().numpy(), "blue", label= r'$(1^T K_0(x,2))_1$')
axs[2,2].plot(x_illu_tor.squeeze().detach().numpy(), vv_y_illu_at2_2order[:,1,:].squeeze().detach().numpy(), "red",  label= r'$(1^T K_0(x,2))_2$')
axs[2,2].set_ylabel('2nd order polyn. k', fontsize=13, rotation='vertical')
axs[2,2].set_xlabel("x", fontsize=15)
axs[2,2].tick_params(labelsize=15)



axs[0,0].legend(fontsize=12)
axs[0,1].legend(fontsize=12)
axs[0,2].legend(fontsize=12)
axs[1,0].legend(fontsize=12)
axs[1,1].legend(fontsize=12)
axs[1,2].legend(fontsize=12)
axs[2,0].legend(fontsize=12)
axs[2,1].legend(fontsize=12)
axs[2,2].legend(fontsize=12)


fig.suptitle("Matrix-valued Stein reproducing kernel", fontsize = 20)

plt.show()


# fig.savefig('mv_Steinkernel.pdf')
