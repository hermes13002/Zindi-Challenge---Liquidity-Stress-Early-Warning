from sklearn.metrics import log_loss, roc_auc_score

def zindi_score(y_true, y_pred_proba):
    """
    Calculates the multi-metric evaluation for the Zindi competition.
    - Log Loss (60%)
    - ROC-AUC (40%)
    """
    loss = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"--- Model Evaluation ---")
    print(f"Log Loss: {loss:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"------------------------")
    
    return loss, auc
