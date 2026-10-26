from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def append_metrics(result_file, metrics):
    with open(result_file, "a", encoding="utf-8", newline="") as file:
        file.write("\nmetric\tvalue\n")
        for name, value in metrics.items():
            file.write(f"{name}\t{value:.6f}\n")
