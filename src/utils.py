import os


def ensure_directories():
    """
    Create necessary directories if they don't exist.
    """
    dirs = ["data", "models", "app", "notebooks", "src"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def print_metrics(auc):
    """
    Display model AUC score in readable form.
    """
    print("\n==========================")
    print(f" Model Performance Report ")
    print("==========================")
    print(f"AUC Score: {auc:.4f}\n")
