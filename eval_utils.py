from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt

def plot_confusion_matrix(test_y, y_pred, inverse_dict, title="Confusion Matrix"):
    """
    Plot a confusion matrix for the given true and predicted labels.
    """
    cm = confusion_matrix(test_y, y_pred, labels=list(inverse_dict.keys()))
    class_names = [inverse_dict[i] for i in range(len(inverse_dict))]
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title)
    plt.show()

def calculate_metrics(y_true, y_pred):
    """
    Calculate and print accuracy, micro-F1, and macro-F1 scores.
    """
    accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {accuracy}")
    print(f"Micro-F1: {micro_f1}")
    print(f"Macro-F1: {macro_f1}")
    return accuracy, micro_f1, macro_f1