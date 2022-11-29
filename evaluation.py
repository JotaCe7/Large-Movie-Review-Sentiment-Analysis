import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize


def get_performance(predictions, y_test, labels=[1, 0]):
    # Put your code
    accuracy = metrics.accuracy_score(y_test, predictions)
    precision = metrics.precision_score(y_test, predictions)
    recall = metrics.recall_score(y_test, predictions)
    f1_score = metrics.f1_score(y_test, predictions)
    
    report = metrics.precision_recall_fscore_support(y_test, predictions)
    
    cm = metrics.confusion_matrix(y_test, predictions)  # replace
    cm_as_dataframe = pd.DataFrame(data=cm)
    
    print('Model Performance metrics:')
    print('-'*30)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
    print('\nModel Classification report:')
    print('-'*30)
    print(report)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    print(cm_as_dataframe)
    
    return accuracy, precision, recall, f1_score


def plot_roc(model, y_test, features, ax=None, title='Receiver Operating Characteristic (ROC) Curve'):
    # Put your code
    y_proba = model.predict_proba(features)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    roc_auc = metrics.roc_auc_score(y_test, y_proba)

    if ax is None:
      plt.figure(figsize=(10, 5))
      ax = plt
      ax.xlim([0.0, 1.0])
      ax.ylim([0.0, 1.05])
      ax.xlabel('False Positive Rate')
      ax.ylabel('True Positive Rate')
      ax.title(title)
      ax.legend(loc="lower right")
    else:
      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([0.0, 1.05])
      ax.set_xlabel('False Positive Rate')
      ax.set_ylabel('True Positive Rate')
      ax.set_title(title)


    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc})', linewidth=2.5)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.legend(loc="lower right")
    # ax.show()

    return roc_auc

def plot_roc_train_test(model, features_train, y_train, features_test, y_test, sup_title=""):

    roc_auc = []
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
    plt.suptitle(sup_title)
    roc_auc.append(plot_roc(model, y_train, features_train, axes[0], title="Receiver Operating Characteristic (ROC) Curve in Train Set"))
    roc_auc.append(plot_roc(model, y_test, features_test, axes[1], title="Receiver Operating Characteristic (ROC) Curve in Test Set"))
    return roc_auc