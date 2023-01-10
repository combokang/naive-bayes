# %%
from random import seed
from naive_bayes import load_list, load_csv, continuous_column_to_float, ten_bin_discretization, descrete_column_to_int, cross_validation_split, dirichlet_prior_nb
import logging


# %%
log_file = 'log/dirichlet.log'
logging.basicConfig(format='%(message)s',
                    filename=log_file, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info('''=============================================
dirichlet-glass
=============================================''')

# %%
seed(1)
filename = 'ranked_attr/glass.txt'
ranked_attributes = load_list(filename)
filename = 'csv/glass.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    continuous_column_to_float(dataset, i)
    ten_bin_discretization(dataset, i)
# convert class column to integers
descrete_column_to_int(dataset, len(dataset[0])-1)

# evaluate algorithm
k_values = [10]*9
n_folds = 5
folds = cross_validation_split(dataset, n_folds)
ranked_columns, avg_score = dirichlet_prior_nb(
    folds, k_values, ranked_attributes)
