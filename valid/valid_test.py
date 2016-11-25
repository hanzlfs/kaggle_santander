import numpy as np
import ml_metrics as metrics

def apk(actual, predicted, k=7):
    """
    Computes the average precision at k = 7.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of string that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapAtK(actual, predicted, m, k = 7):
	"""
	predicted: list of predicted items string, like ['ind_cco_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_cder_fin_ult1', 'ind_ctma_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctop_fin_ult1']
	actual : []
	"""
