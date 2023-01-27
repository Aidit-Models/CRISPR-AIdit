import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# For Prediction & Evaluation
def evaluation(y_train, y_pred):
    from sklearn.metrics import mean_squared_error
    # spearman
    spearman = pd.Series(y_pred).corr(pd.Series(y_train), method='spearman')
    # pearson
    pearson = pd.Series(y_pred).corr(pd.Series(y_train), method='pearson')
    # MSE
    mse = mean_squared_error(y_train, y_pred)
    return (spearman, pearson, mse)


# plot auc
def plot_ROC(data, ytrue_col, ypred_col_dict, savefig_path, curve='auc'):
    ytrue = data[ytrue_col]
    # ================================ Ploting ====================================
    colors = ['red', 'royalblue', 'darkorange', 'lightgreen', 'palevioletred', 'teal',
              'maroon', 'indigo', 'darkorchid', 'mediumorchid', 'thistle', 'pink', 'blueviolet',
              'plum', 'violet', 'purple', 'm', 'lightseagreen', 'magenta',
              'orchid', 'chartreuse', 'deeppink', 'hotpink']
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    if len(ypred_col_dict) != 0:
        xlabel, ylabel = 'x', 'y'
        i = 0
        for label, ypred_col in ypred_col_dict.items():
            ypred = data[ypred_col]
            if curve == 'auc':
                score = roc_auc_score(ytrue, ypred)
                score = ', AUC=%s' % (round(score, 3))
                fpr, tpr, thresholds = roc_curve(ytrue, ypred, pos_label=1)  # pos_label=1，表示值为1的实际值为正样本
                x, y = fpr, tpr
                xlabel, ylabel = 'False Postive Rate', 'True Positive Rate'
            else:
                precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
                x, y = precision, recall
                score = auc(recall, precision)
                score = ', PR-AUC=%s' % (round(score, 3))
                xlabel, ylabel = 'Recall', 'Precision'
                # plot
            plt.plot(x, y, colors[i], label=label + score, linewidth=0.7)
            i += 1
        # title
        if curve == 'auc':
            plt.ylim(0.5, 1.025)
        plt.xlabel(xlabel, fontsize=12, weight='bold')
        plt.ylabel(ylabel, fontsize=12, weight='bold')
        plt.legend(prop={'weight': 'bold', 'size': 6})
        plt.savefig(savefig_path, dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()