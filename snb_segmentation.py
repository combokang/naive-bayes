# %%
from random import seed
from naive_bayes import load_csv, continuous_column_to_float, ten_bin_discretization, descrete_column_to_int, cross_validation_split, snb
import logging


# %%
log_file = 'log/snb.log'
logging.basicConfig(format='%(message)s',
                    filename=log_file, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info('''=============================================
snb-segmentation
=============================================''')


# %%
# data preprocessing
seed(1)
filename = 'csv/segmentation.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    continuous_column_to_float(dataset, i)
    ten_bin_discretization(dataset, i)
# convert class column to integers
descrete_column_to_int(dataset, len(dataset[0])-1)

# evaluate algorithm
k_values = [10]*19
n_folds = 5
folds = cross_validation_split(dataset, n_folds)
ranked_columns, avg_score = snb(folds, k_values)

with open('ranked_attr/segmentation.txt', 'w') as file:
    file.write(', '.join(str(e) for e in ranked_columns))
