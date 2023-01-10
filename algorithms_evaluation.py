# Glass
LENGTHS = [43, 43, 43, 43, 42]
# ACCURACIES_1 = [0.62791, 0.46512, 0.65116, 0.51163, 0.57143]
# ACCURACIES_2 = [0.65116, 0.46512, 0.69767, 0.51163, 0.54762]
# ACCURACIES_3 = [0.67442, 0.48837, 0.69767, 0.69767, 0.59524]
# Hepatits
# LENGTHS = [31, 31, 31, 31, 31]
# ACCURACIES_1 = [0.74194, 0.80645, 0.80645, 0.90323, 0.83871]
# ACCURACIES_2 = [0.74194, 0.90323, 0.80645, 0.93548, 0.90323]
# ACCURACIES_3 = [0.74194, 0.90323, 0.83871, 0.90323, 0.90323]
# Segmentation
# LENGTHS = [420, 420, 420, 420, 420]
ACCURACIES_1 = [0.89048, 0.86429, 0.89286, 0.89762, 0.87857]
ACCURACIES_2 = [0.90000, 0.89524, 0.90000, 0.91429, 0.89286]
ACCURACIES_3 = [0.89762, 0.88810, 0.89762, 0.90476, 0.89524]

FOLD_SPLIT = 5

comparing_ojbect_1 = ACCURACIES_1
comparing_ojbect_2 = ACCURACIES_3

print("Algorithm 1")
for i, (m, p) in enumerate(zip(LENGTHS, comparing_ojbect_1)):
    print(f'''fold {i+1}
m = {m}, p = {p}
m*p = {m*p}
m*(1-p) = {m*(1-p)}
=============================================''')

print("Algorithm 2")
for i, (m, p) in enumerate(zip(LENGTHS, comparing_ojbect_2)):
    print(f'''fold {i+1}
m = {m}, p = {p}
m*p = {m*p}
m*(1-p) = {m*(1-p)}
=============================================''')

delta = list()
for i, j in zip(comparing_ojbect_1, comparing_ojbect_2):
    delta.append(j-i)
delta_bar = sum(delta)/FOLD_SPLIT
print(f"average delta: {delta_bar}")

sum = 0
for i in delta:
    sum += (i-delta_bar)**2
std2 = sum/(FOLD_SPLIT-1)
print(f"squred standard deviation: {std2}")

t = delta_bar/(std2/FOLD_SPLIT)**0.5
print(f"test statistics t-value: {t}")
