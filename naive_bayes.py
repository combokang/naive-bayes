# %%
from csv import reader
from random import randrange
import logging
import copy


# 從txt檔讀取整數list
def load_list(filename):
    with open(filename, 'r') as file:
        line = file.readline()
        lst = [int(i) for i in line.split(',')]
    return lst


# 載入CSV檔
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# 處理連續屬性
def continuous_column_to_float(dataset, column):
    for row in dataset:
        if row[column] == '?':
            row[column] = None
        else:
            row[column] = float(row[column].strip())


# 處理離散屬性
def descrete_column_to_int(dataset, column):
    class_values = list()
    for row in dataset:
        if row[column] == '?':
            row[column] = None
        elif row[column] not in class_values:
            class_values.append(row[column])
    lookup = dict()
    for i, value in enumerate(class_values):
        lookup[value] = i
    for row in dataset:
        if row[column] is not None:
            row[column] = lookup[row[column]]
    return lookup


# Ten-bin離散化
def ten_bin_discretization(dataset, column):
    values = [row[column] for row in dataset if row[column] is not None]
    max_value, min_value = max(values), min(values)
    width = (max_value-min_value)/10
    # 離散化成0~9
    for row in dataset:
        if row[column] is None:
            continue
        elif width == 0:
            row[column] = 0
        else:
            row[column] = int((row[column]-min_value)//width)
            if row[column] == 10:
                row[column] -= 1


# K-folds分割
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    for fold in dataset_split:
        if len(dataset_copy) == 0:
            break
        index = randrange(len(dataset_copy))
        fold.append(dataset_copy.pop(index))
    return dataset_split


# 進行欄位排序
def snb(folds, k_values):
    rest_columns = [i for i in range(len(k_values))]
    ranked_columns = list()
    while len(ranked_columns) < len(k_values):
        best_column, best_score = None, 0
        for i in rest_columns:
            columns_for_snb = ranked_columns + [i]
            avg_score = evaluate_algorithm(
                folds, k_values, _type='SNB', columns_for_snb=columns_for_snb)
            if best_column is None or avg_score > best_score:
                best_score = avg_score
                best_column = i
        logging.info(f'Best Attribute: Column {best_column}')
        ranked_columns.append(best_column)
        rest_columns.remove(best_column)
    lengths = [str(len(fold)) for fold in folds]
    logging.info(f'''=============================================
Data of instances for folds: {', '.join(lengths)}
Attribute Ranking: {ranked_columns}
Mean Accuracy (With All Attributes): {(best_score):.3f}%
=============================================''')
    return ranked_columns, best_score


# Dirichlet Prior參數尋找
def dirichlet_prior_nb(folds, k_values, ranked_attributes):
    best_alphas = dict()
    for i in ranked_attributes:
        best_alpha, best_score = None, 0
        for alpha in range(1, 51):
            test_alphas = best_alphas.copy()
            test_alphas[i] = alpha
            avg_score = evaluate_algorithm(
                folds, k_values, _type='Dirichlet', test_paras=test_alphas)
            if best_alpha is None or avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
        best_alphas[i] = best_alpha
        logging.info(f'Best Alpha for Column {i}: {best_alpha}')
    lengths = [str(len(fold)) for fold in folds]
    logging.info(f'''=============================================
Data of instances for folds: {', '.join(lengths)}
Best Alphas: {best_alphas}
Mean Accuracy: {(best_score):.3f}%
=============================================''')
    return best_alphas, best_score


# Generalized Dirichlet Prior參數尋找
def gdirichlet_prior_nb(folds, k_values, ranked_attributes):
    best_paras = dict()
    for i in ranked_attributes:
        k_gd = k_values[i]-1
        default_alphas = [1]*(k_gd)    # alphai=1
        default_betas = [(k_gd-j)*alpha for j,
                         alpha in enumerate(default_alphas)]  # betai=(k-i+1)alphai
        best_paras[i] = dict()
        best_paras[i]['alpha'] = default_alphas
        best_paras[i]['beta'] = default_betas
        for alpha_index in range(k_gd):
            best_alpha, best_score = None, 0
            test_paras = best_paras.copy()
            for alpha in range(1, 51):
                test_paras[i]['alpha'][alpha_index] = alpha
                beta = (k_gd-alpha_index)*alpha  # betai=(k-i+1)alphai
                test_paras[i]['beta'][alpha_index] = beta
                avg_score = evaluate_algorithm(
                    folds, k_values, _type='gDirichlet', test_paras=test_paras)
                if best_alpha is None or avg_score > best_score:
                    best_score = avg_score
                    best_alpha = alpha
                    best_beta = beta
            best_paras[i]['alpha'][alpha_index] = best_alpha
            best_paras[i]['beta'][alpha_index] = best_beta
            logging.info(
                f'Best Alpha {alpha_index+1} for Column {i}: {best_alpha}')
        logging.info(f'Best GD Parameters for Column {i}: ({best_paras})')
    lengths = [str(len(fold)) for fold in folds]
    logging.info(f'''=============================================
Data of instances for folds: {', '.join(lengths)}
Best GD Parameters: {best_paras}
Mean Accuracy: {(best_score):.3f}%
=============================================''')
    return best_paras, best_score


# Cross validation評估
def evaluate_algorithm(folds, k_values, _type='Normal', columns_for_snb=None, test_paras=None):
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = naive_bayes(train_set, test_set,
                                k_values, _type, columns_for_snb, test_paras)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    avg_score = sum(scores)/float(len(scores))
    formated_scores = [f'{i:.3f}%' for i in scores]
    stream = ''
    if _type == 'SNB':
        stream += f'Attributes: {columns_for_snb}\n'
    elif _type == 'Dirichlet':
        stream += f'Alphas:{test_paras}\n'
    elif _type == 'gDirichlet':
        stream += f'Parameters:{test_paras}\n'
    stream += f'Accuracies: {", ".join(formated_scores)}, Mean Accuracy: {(avg_score):.3f}%'
    logging.info(stream)
    return avg_score


def naive_bayes(train, test, k_values, _type, columns_for_snb, test_paras):
    class_probabilities, attr_probabilities = training(
        train, k_values, _type, test_paras)
    predictions = list()
    for row in test:
        output = predict(row, k_values,
                         class_probabilities, attr_probabilities, columns_for_snb)
        predictions.append(output)
    return(predictions)


# 訓練
def training(train, k_values, _type, test_paras):
    separated = separate_by_class(train)
    # 計算P(C)
    class_probabilities = dict()
    total_rows = len(train)
    for class_value in separated:
        class_probabilities[class_value] = len(
            separated[class_value])/total_rows
    # 計算P(Xi|C)
    possible_valuecount = dict()  # 保存可能值出現次數
    for class_value in separated:
        # 初始化dictionary
        possible_valuecount[class_value] = {
            i: [0]*k_values[i] for i in range(len(k_values))}
        # 每個欄位可能值計數
        for row in separated[class_value]:
            for i in range(len(k_values)):
                if row[i] is not None:
                    possible_valuecount[class_value][i][row[i]] += 1
    # 計算每個欄位每個可能值機率
    attr_priors = copy.deepcopy(possible_valuecount)    # 保存後驗機率
    for class_value in separated:
        for i in range(len(k_values)):
            ys = possible_valuecount[class_value][i]
            total = sum(ys)
            for j in range(k_values[i]):
                if test_paras is not None and i in test_paras:
                    if _type == 'Dirichlet':    # Dirichlect
                        alpha = test_paras[i]
                        attr_priors[class_value][i][j] = (
                            ys[j]+alpha)/(total+k_values[i]*alpha)
                    elif _type == 'gDirichlet':    # generalized Dirichlect
                        alphas = test_paras[i]['alpha']
                        betas = test_paras[i]['beta']
                        k_gd = k_values[i]-1
                        if j != k_gd:
                            temp = (alphas[j]+ys[j]) / \
                                (alphas[j]+ys[j]+betas[j]+sum(ys[j+1:]))
                            for k in range(j):
                                temp *= (betas[k]+sum(ys[k+1:]))/(
                                    alphas[k]+ys[k]+betas[k]+sum(ys[k+1:]))
                        else:
                            temp = 1
                            for k in range(k_gd):
                                temp *= (betas[k]+sum(ys[k+1:]))/(
                                    alphas[k]+ys[k]+betas[k]+sum(ys[k+1:]))
                        attr_priors[class_value][i][j] = temp
                else:   # Laplace's
                    attr_priors[class_value][i][j] = (
                        possible_valuecount[class_value][i][j]+1)/(total+k_values[i])
    return class_probabilities, attr_priors


# 將資料集依class區分
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        attributes = dataset[i]
        class_value = attributes[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(attributes)
    return separated


# 預測
def predict(row, k_values, class_probabilities, attr_probabilities, columns_for_snb):
    probabilities = calculate_class_probabilities(
        row, k_values, class_probabilities, attr_probabilities, columns_for_snb)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# 計算單一資料屬於各個class的機率
def calculate_class_probabilities(row, k_values, class_probabilities, attr_probabilities, columns_for_snb):
    probabilities = dict()
    for class_value in class_probabilities:
        probabilities[class_value] = class_probabilities[class_value]
        if columns_for_snb is None:  # 沒有用snb的時候用全部的欄位計算
            columns_for_snb = range(len(k_values))
        for i in columns_for_snb:  # 計算每個屬性P(Xi|C)
            if row[i] is not None:
                p = attr_probabilities[class_value][i][row[i]]
                probabilities[class_value] *= p
    return probabilities


# 計算正確率
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
