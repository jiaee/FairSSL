from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score

metrics = {
    'sel': selection_rate,
    'tpr': true_positive_rate,
    'fpr': false_positive_rate,
    'accuracy': accuracy_score
}

def demographic_parity(gt, pred, sf_data):

    mf = MetricFrame(y_true=gt, y_pred=pred, sensitive_features=sf_data, metrics=metrics)
    val1 = mf.by_group['sel'][0]    # female = 0
    val2 = mf.by_group['sel'][1]    # male = 1
    # return min (val1/val2, val2/val1)
    if val2 == 0:
        return float('inf')

    return val1/val2
 

def equal_opportunity(gt, pred, sf_data):

    val1 = sum([1 for i in range(len(pred)) if pred[i]==1 and sf_data[i]==0])   # female = 0
    val2 = sum([1 for i in range(len(pred)) if pred[i]==1 and sf_data[i]==1])   # male = 1

    # print(val1, val2)
    # return min(val1/val2, val2/val1)

    if val2 == 0:
        # return infinity
        return float('inf')

    return val1/val2
 

def equalized_odds(gt, pred, sf_data):

    mf = MetricFrame(y_true=gt, y_pred=pred, sensitive_features=sf_data, metrics=metrics)
    val1_tpr = mf.by_group['tpr'][0]
    val1_fpr = mf.by_group['fpr'][0]
    val2_tpr = mf.by_group['tpr'][1]
    val2_fpr = mf.by_group['fpr'][1]
    # return min(max(val1_fpr/val2_fpr, val1_tpr/val2_tpr), max(val2_fpr/val1_fpr, val2_tpr/val1_tpr))

    if val2_fpr == 0 or val2_tpr == 0:
        return float('inf')

    return max(val1_fpr/val2_fpr, val1_tpr/val2_tpr)
 

def equal_accuracy(gt, pred, sf_data):

    mf = MetricFrame(y_true=gt, y_pred=pred, sensitive_features=sf_data, metrics=metrics)
    val1 = mf.by_group['accuracy'][0]
    val2 = mf.by_group['accuracy'][1]
    # return min(val1/val2, val2/val1)

    if val2 == 0:
        return float('inf')
    
    return val1/val2

def calculate_metrics(y_pred, y_labels, gender_list, y_scores=None, linearprobing=False):
    """
    Calculates fairness and performance metrics.
    
    Args:
        y_pred (list or np.array): Predicted labels.
        y_labels (list or np.array): True labels.
        gender_list (list or np.array): Sensitive attribute (e.g., 0 for Female, 1 for Male).
        linearprobing (bool): Flag to control print statements.
        
    Returns:
        tuple: SP, EOpp, EOdd, EAcc, Acc, Prec, Sens, Spec, F1, AUC
    """
    # Calculate Fairness Metrics
    SP = demographic_parity(y_labels, y_pred, gender_list)
    EOpp = equal_opportunity(y_labels, y_pred, gender_list)
    EOdd = equalized_odds(y_labels, y_pred, gender_list)
    EAcc = equal_accuracy(y_labels, y_pred, gender_list)
    
    # Calculate Performance Metrics
    cm = confusion_matrix(y_labels, y_pred)
    tn, fp, fn, tp = cm.ravel()

    Sens = tp / (tp + fn)
    Spec = tn / (fp + tn)
    uar = (Spec + Sens)/2.0
    Acc = (tp + tn) / (tp + tn + fp + fn)
    
    if (tp + fp) > 0:
        Prec = tp / (tp + fp)
    else:
        Prec = float('inf')

    if (Prec + Sens) > 0:
        F1 = 2 * Prec * Sens / (Prec + Sens)
    else:
        F1 = float('inf') 
    
    # Compute ROC AUC if scores provided
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_labels, y_scores)
        except ValueError:
            # fallback if only one class present
            auc = float('nan')
    else:
        auc = float('nan')


    if not linearprobing:
        # Print the total number of each category
        total_males_with_mdd = sum((1 for i in range(len(y_pred)) if gender_list[i]==1 and y_labels[i]==1))
        total_females_with_mdd = sum((1 for i in range(len(y_pred)) if gender_list[i]==0 and y_labels[i]==1))
        total_males_with_hc = sum((1 for i in range(len(y_pred)) if gender_list[i]==1 and y_labels[i]==0))
        total_females_with_hc = sum((1 for i in range(len(y_pred)) if gender_list[i]==0 and y_labels[i]==0))
        
        print(f"Total Males with MDD: {total_males_with_mdd}")
        print(f"Total Females with MDD: {total_females_with_mdd}")
        print(f"Total Males with HC: {total_males_with_hc}")
        print(f"Total Females with HC: {total_females_with_hc}")

        # Now compute confusion matrices by gender
        male_indices = [i for i, g in enumerate(gender_list) if g == 1]
        male_labels = [y_labels[i] for i in male_indices]
        male_preds = [y_pred[i] for i in male_indices]
        male_cm = confusion_matrix(male_labels, male_preds, labels=[0,1])
        male_tn, male_fp, male_fn, male_tp = male_cm.ravel()

        female_indices = [i for i, g in enumerate(gender_list) if g == 0]
        female_labels = [y_labels[i] for i in female_indices]
        female_preds = [y_pred[i] for i in female_indices]
        female_cm = confusion_matrix(female_labels, female_preds, labels=[0,1])
        female_tn, female_fp, female_fn, female_tp = female_cm.ravel()

        print("---- MALE GROUP (Gender=1) ----")
        print(f"Males with MDD: True Positives = {male_tp}, False Negatives = {male_fn}")
        print(f"Males with HC:  True Negatives  = {male_tn}, False Positives = {male_fp}\n")

        print("---- FEMALE GROUP (Gender=0) ----")
        print(f"Females with MDD: True Positives = {female_tp}, False Negatives = {female_fn}")
        print(f"Females with HC:  True Negatives  = {female_tn}, False Positives = {female_fp}\n")

        
    return SP, EOpp, EOdd, EAcc, Acc, Prec, Sens, Spec, F1, uar, auc
