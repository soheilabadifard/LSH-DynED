from itertools import combinations
import numpy as np


def mmr(indx, diversity_matrix, accuracy_scores, lmd, to_select):
    """
    Calculates the Maximal Marginal Relevance (MMR) scores for a given set of classifiers.

    Parameters
    ----------
    indx : list
        A list of indices of the classifiers.
    diversity_matrix : ndarray
        A square matrix where each element (i, j) is the diversity score between the i-th and j-th classifier.
    accuracy_scores : list
        A list of accuracy scores for each classifier.
    lmd: float
        The lambda parameter is used in the MMR calculation. It determines the trade-off between accuracy and diversity.
    to_select : int
        The number of classifiers to select.

    Returns
    -------
    s: list
        A list of indices of the selected classifiers.
    score: float
        The total MMR score for the selected classifiers.
    """
    len_classifiers = len(accuracy_scores)
    s = []
    aux = []
    score = 0
    while len(s) < to_select:
        if not s:
            s.append(indx[accuracy_scores.index(max(accuracy_scores))])
            aux.append(accuracy_scores.index(max(accuracy_scores)))
            score += max(accuracy_scores)
        else:
            classifier_index = [i for i in range(len_classifiers) if i not in aux]
            mmr_list = []
            for i in classifier_index:
                auxi = [diversity_matrix[i, j] for j in aux]
                mmr_list.append([i, lmd * accuracy_scores[i] - (1 - lmd) * max(auxi)])
            s.append(indx[max(mmr_list, key=lambda i: i[1])[0]])
            aux.append(max(mmr_list, key=lambda i: i[1])[0])
            score += max(mmr_list, key=lambda i: i[1])[1]
        if len(s) == len_classifiers:
            break
    return s, score

def kappa_metric(predict_queue):
    """
    Calculates the Cohen's kappa metric for a given queue of predictions.

    Parameters
    ----------
    predict_queue : list
        The list of predictions for which the kappa metric is to be calculated.

    Returns
    -------
    kappa_matrix : ndarray
        A square matrix where each element (i, j) is the kappa metric between the i-th and j-th prediction in the queue.
    """
    min_length = len(predict_queue[0])
    kappa_matrix = np.zeros((len(predict_queue), len(predict_queue)), dtype=float)

    for i, j in combinations(range(len(predict_queue)), 2):
        both_correct = np.count_nonzero(np.array(predict_queue[i]) & np.array(predict_queue[j]))
        both_incorrect = min_length - np.count_nonzero(
            (np.array(predict_queue[i]) | np.array(predict_queue[j])))
        fcorrect_sincorrect = np.count_nonzero(np.array(predict_queue[i]) & ~np.array(predict_queue[j]))
        fincorrect_scorrect = np.count_nonzero(~np.array(predict_queue[i]) & np.array(predict_queue[j]))

        p_observed = (both_correct + both_incorrect) / min_length
        p_expected = ((both_correct + fcorrect_sincorrect) / min_length) * (
                both_correct + fincorrect_scorrect) / min_length + \
                     ((both_incorrect + fincorrect_scorrect) / min_length) * (
                             fcorrect_sincorrect + both_incorrect) / min_length
        if p_expected == 1:
            p_expected = 1 - np.finfo(float).eps
        kappa_matrix[i, j] = kappa_matrix[j, i] = (p_observed - p_expected) / (1 - p_expected)

    return kappa_matrix
