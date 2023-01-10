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
dirichlet-hepatitis
=============================================''')

# %%
# data preprocessing
seed(1)
filename = 'ranked_attr/hepatitis.txt'
ranked_attributes = load_list(filename)
filename = 'csv/hepatitis.csv'
dataset = load_csv(filename)
for i in [0, 13, 14, 15, 16, 17]:
    continuous_column_to_float(dataset, i)
    ten_bin_discretization(dataset, i)
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]:
    descrete_column_to_int(dataset, i)
# convert class column to integers
descrete_column_to_int(dataset, len(dataset[0])-1)

# evaluate algorithm
k_values = [10]+[2]*12+[10]*5+[2]
n_folds = 5
folds = cross_validation_split(dataset, n_folds)
ranked_columns, avg_score = dirichlet_prior_nb(
    folds, k_values, ranked_attributes)
print(f'Best Alphas: {ranked_columns}')
print(f'Mean Accuracy: {(avg_score):.3f}%')
