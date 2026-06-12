import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

def find_optimal_threshold_and_plot_cm(labels_np, probs_np, num_classes, class_names=None, model_name=''):
    out_dir = "cm_plots"
    os.makedirs(out_dir, exist_ok=True)
    cm_all = []
    
    for i in range(num_classes):
        # Ignore class if it has only one class present in true labels 
        if len(np.unique(labels_np[:, i])) < 2:
            print(f"Skipping Class {i} because it has only one class in actual labels.")
            continue
            
        fpr, tpr, thresholds = roc_curve(labels_np[:, i], probs_np[:, i])
        # Youden's J statistic
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        class_name = class_names[i] if (class_names and i < len(class_names)) else f"Class_{i}"
        print(f"Class {class_name} Optimal Threshold: {optimal_threshold:.4f}")
        
        # Predict using optimal threshold
        preds_opt = (probs_np[:, i] >= optimal_threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(labels_np[:, i], preds_opt)
        cm_all.append(cm)
        
        # Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'{model_name}\nConfusion Matrix - {class_name}\nOptimal Threshold: {optimal_threshold:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        
        save_path = os.path.join(out_dir, f'cm_{model_name}_{class_name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix plot to {save_path}")

    return cm_all