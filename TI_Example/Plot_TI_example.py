import pickle

# Extract Saved Outputs
with open('TI_example_all_data.pkl', 'rb') as input:
    I_svCV = pickle.load(input)
    print(I_svCV)

    I_vvCV = pickle.load(input)
    print(I_vvCV)



# Van der Poll Oscillator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
def vdp(t, z):
    x, y = z
    return [y, mu * (1 - x ** 2) * y - x]

a, b = 0, 10  # time period
mus = [1]
styles =['-']
t = np.linspace(a, b, 101)

for mu, style in zip(mus, styles):
    sol = solve_ivp(vdp, [a, b], [1, 0], t_eval=t)



# Add some obserations at  {0, 1, ... 10}
raw_time_indices = torch.tensor([0,1,2,3,4,5,6,7,8,9,10])
time_indices = raw_time_indices * 10
sols_y0 = torch.Tensor(sol.y[0])[time_indices]
sols_y0.size()
torch.manual_seed(0)
obs_ =  sols_y0 + torch.randn(11) * 0.1
obs_ - sols_y0



# Visualization
fig = plt.figure()
fig.set_figwidth(12)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2 )
ax3 = fig.add_subplot(1, 3, 3, sharex = ax2, sharey = ax2)

ax1.plot(sol.t, sol.y[0], ls='-', color = 'black')
ax1.scatter(raw_time_indices, obs_, color = 'red')
ax1.legend([r'$\theta={1}$'], fontsize = 14)

ax2.boxplot(I_svCV.detach().numpy())
ax3.boxplot(I_vvCV.detach().numpy())
x_ticks_labels = ['20','40', '60', '80']

# Set number of ticks for x-axis
ax2.set_xticks([1,2,3, 4])
ax3.set_xticks([1,2,3, 4])

# Set ticks labels for x-axis
ax1.tick_params(labelsize=13)
ax2.tick_params(labelsize=13)
ax3.tick_params(labelsize=13)
ax2.set_xticklabels(x_ticks_labels, fontsize=13) #rotation='vertical'
ax3.set_xticklabels(x_ticks_labels, fontsize=13) #rotation='vertical'

ax2.axhline(y=25.58,c="black", ls='--', linewidth=0.5,zorder=0, label = 'CF')
ax3.axhline(y=25.58,c="black", ls='--', linewidth=0.5,zorder=0, label = 'CF')

ax1.set_title("van der Poll oscillator", fontsize = 16)
ax2.set_title("Kernel-based CVs", fontsize = 16)
ax3.set_title("Kernel-based vv-CVs", fontsize = 16)

ax1.set_ylabel('x', fontsize = 15)
ax2.set_ylabel('Model Evidence', fontsize=15)
ax2.set_ylabel('Model Evidence', fontsize=15)

ax1.set_xlabel('Time', fontsize = 15)
ax2.set_xlabel('Sample Size', fontsize = 15)
ax3.set_xlabel('Sample Size', fontsize = 15)

plt.show()

# fig.savefig('filename.pdf')





