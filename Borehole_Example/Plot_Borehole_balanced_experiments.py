import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Extract Saved Outputs
my_vv_CV_DF = pd.read_pickle("../data/Balanced_Borehole_Example_PDframe.pkl")

# sns.set_style(style='white')
# sns.set_style(style='whitegrid')
g=sns.catplot(x="method_idx",
              y="cv_est",
              hue="sample_size",
              col="func_idx",
              data=my_vv_CV_DF,
              kind="box",
              height=4,
              aspect=.7,
              palette="Set2",
              medianprops={'color':'blue'},
              showmeans=True,
              meanprops={"marker": "+",
                         "linestyle": "--",
                         "color": "red",
                         "markeredgecolor": "red",
                         "markersize": "10"})


(g.set_axis_labels("", "Abs. Err.")
  .set_titles("{col_name}")
  .despine(left=True)
 )
g._legend.remove()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()







