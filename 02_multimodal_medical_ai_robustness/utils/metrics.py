from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)



def calculate_metrics(
        targets,
        predictions,
        probabilities=None
):

    results={}


    results["accuracy"] = accuracy_score(
        targets,
        predictions
    )


    results["f1"] = f1_score(
        targets,
        predictions
    )


    results["precision"] = precision_score(
        targets,
        predictions
    )


    results["recall"] = recall_score(
        targets,
        predictions
    )


    if probabilities is not None:

        results["auc"] = roc_auc_score(
            targets,
            probabilities
        )


    return results
