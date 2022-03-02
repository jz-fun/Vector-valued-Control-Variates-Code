# Vector-valued Control Variates
Code release for the paper [Vector-valued Control Variates](https://arxiv.org/abs/2109.08944).

## Requirements
We run code on the basis of _Python 3.7_, _Pycharm_ and a Mac Laptop with Intel chip.
```
matplotlib==3.5.1
scipy==1.8.0
torch==1.10.2
numpy==1.22.2
seaborn==0.11.2
pandas==1.4.1
```
If you find difficulties to install _torch_ using the _requirements.txt_, please go to the website of pytorch and install a version suitable for your computer.


## Use VVCVs for your own data
```python
# An example
# Step 1. Initialize vvCV model class
my_vvCV_model = VV_CV_vectorvaluedfuncs_model_MDMP(vv_cv_objective, \
                                                   prior_kernel, base_kernel, \
                                                   Xs_tensor, Ys_tensor, scores_Tensor)
        """
        :param vv_cv_objective:  an objective class. Here should be 'penalized_ls_objective_vectorvaluedfunc_MDMP'
        :param prior_kernel:   a class. Here should be 'stein_matrix_valued_kernel'
        :param base_kernel:    a class; since we use K = B k, so here this class is scalar-valued kernel class, e.g, base_kernel_2
        :param Xs_tensor:     3d tensor, T * m * d when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param Ys_tensor:     3d tensor, T * m * 1 when assuming all datasets have m points, i.e. m = m_1 =... =m_T
        :param scores_Tensor: 4d Tensor, T * m * T * d ; That is, we have T tasks, each has a dataset with sample size m. For each instance, we need a 2d Tensor of size T*d.
        """

# Step 2. Tune kernel hyper-parameters
my_vvCV_model.do_tune_kernelparams_negmllk(batch_size_tune, flag_if_use_medianheuristic, beta_cstkernel, lr, epochs, verbose=False)

# Step 3. Optimize model parameters
my_vvCV_model.do_optimize_vv_CV(self, regularizer_const, regularizer_const_FB, batch_size, lr, epochs, verbose=False)

# Step 4. Return simplified CV estimates
my_vvCV_model.saved_BQ_est
```



## Reproducing Experiments
* The code of _Vector-valued Control Variates_ (vvCVs) is in the folder `./VVCVs-Code/TI_Example/src_vvCV_MDMP/`. This includes the most generalised version of vvCVs used in our paper, i.e., when all distributions and integrands are different.

    * A simpler case is when we only have one target distribution and mulitple integrands, of which code is in the folder `./VVCVs-Code/Step_Function/src_vvCV_MD1P/`.
  
* To reproduce the results of _South function example_, run `./VVCVs-Code/South_Function/Plot_South_function_experiments.py` (this will use the data stored in `./VVCVs-Code/South_function_pdframe_saved.pkl`).

   * If you want to rerun the whole experiments, save your own data and get the corresponding plot, read (uncomment the code of storing data) and run `./VVCVs-Code/South_Function/South_function_experiments.py` and then run `./VVCVs-Code/South_Function/Plot_South_function_experiments.py`

* To reproduce the results of _Computation of the Model Evidence for Dynamical Systems_, run `./VVCVs-Code/TI_Example/Plot_TI_example.py` (this will use the data stored in `./VVCVs-Code/TI_example_all_data.pkl`).

   * If you want to rerun the whole experiments, save your own data and get the corresponding plot, read (uncomment the code of storing data) and run `./VVCVs-Code/TI_Example/Thermodynamic_Integration_experiments.py` and then run `./VVCVs-Code/TI_Example/Plot_TI_example.py`

* For the experiments of _step function example_ and _Borehole function example_, the steps are similar.



## Citation
If you use the code, please cite our paper.
```bibtex
@article{sun2021vvCVs,
  title={Vector-Valued Control Variates},
  author={Sun, Zhuo and Barp, Alessandro and Briol, Fran{\c{c}}ois-Xavier},
  journal={arXiv preprint arXiv:2109.08944},
  year={2021}
}
```



## Acknowledgement
- The samples of _Computation of the Model Evidence for Dynamical Systems_ are provided by Chris J. Oates, which are the ones used in [Control functionals for Monte Carlo integration](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/rssb.12185).
