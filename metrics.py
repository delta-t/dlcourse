def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    # precision = 0
    # recall = 0
    # accuracy = 0
    # f1 = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] and ground_truth[i]:
            tp += 1
        if not prediction[i] and ground_truth[i]:
            fn += 1
        if prediction[i] and not ground_truth[i]:
            fp += 1
        if not prediction[i] and not ground_truth[i]:
            tn += 1

#     print((tp + fp + fn + tn) == len(ground_truth))
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    # TODO: Implement computing accuracy
    
    return (prediction == ground_truth).sum() / len(ground_truth)
