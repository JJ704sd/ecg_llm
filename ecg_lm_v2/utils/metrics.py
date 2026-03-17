import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class ECGMetrics:
    def __init__(self, class_names: Optional[List[str]] = None):
        if class_names is None:
            self.class_names = [
                "Normal", "AFib", "AFL", "SVT", "VT", 
                "PVC", "PAC", "LBBB", "RBBB", "Brady", "Tachy", "MI"
            ]
        else:
            self.class_names = class_names
            
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def update(self, preds: np.ndarray, targets: np.ndarray, probs: Optional[np.ndarray] = None):
        self.predictions.extend(preds.flatten())
        self.targets.extend(targets.flatten())
        if probs is not None:
            self.probabilities.append(probs)
            
    def compute(self) -> Dict[str, float]:
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        metrics['precision_macro'] = precision_score(
            targets, predictions, average='macro', zero_division=0
        )
        metrics['recall_macro'] = recall_score(
            targets, predictions, average='macro', zero_division=0
        )
        metrics['f1_macro'] = f1_score(
            targets, predictions, average='macro', zero_division=0
        )
        
        metrics['precision_weighted'] = precision_score(
            targets, predictions, average='weighted', zero_division=0
        )
        metrics['recall_weighted'] = recall_score(
            targets, predictions, average='weighted', zero_division=0
        )
        metrics['f1_weighted'] = f1_score(
            targets, predictions, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        if len(self.probabilities) > 0:
            all_probs = np.vstack(self.probabilities)
            try:
                metrics['auc_macro'] = roc_auc_score(
                    targets, all_probs, average='macro', multi_class='ovr'
                )
            except ValueError:
                metrics['auc_macro'] = 0.0
                
        per_class_metrics = self._compute_per_class_metrics(predictions, targets)
        metrics['per_class'] = per_class_metrics
        
        return metrics
    
    def _compute_per_class_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict:
        per_class = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_targets = (targets == class_idx).astype(int)
            class_preds = (predictions == class_idx).astype(int)
            
            tp = ((class_preds == 1) & (class_targets == 1)).sum()
            fp = ((class_preds == 1) & (class_targets == 0)).sum()
            fn = ((class_preds == 0) & (class_targets == 1)).sum()
            tn = ((class_preds == 0) & (class_targets == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            sensitivity = recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            per_class[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'support': int(class_targets.sum())
            }
            
        return per_class
    
    def get_classification_report(self) -> str:
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return classification_report(
            targets, predictions, 
            target_names=self.class_names,
            digits=4
        )
    
    def save_metrics(self, save_path: str):
        metrics = self.compute()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
        print(f"Metrics saved to {save_path}")
        
    def print_summary(self):
        metrics = self.compute()
        
        print("\n" + "=" * 60)
        print("                    模型评估结果")
        print("=" * 60)
        
        print(f"\n【总体指标】")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
        print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
        
        print(f"\n【加权指标】")
        print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted):   {metrics['recall_weighted']:.4f}")
        print(f"  F1 (weighted):       {metrics['f1_weighted']:.4f}")
        
        if 'auc_macro' in metrics:
            print(f"\n  AUC (macro):      {metrics['auc_macro']:.4f}")
            
        print(f"\n【各类别指标】")
        print("-" * 80)
        print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-" * 80)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<12} {class_metrics['precision']:<12.4f} "
                  f"{class_metrics['recall']:<12.4f} {class_metrics['f1']:<12.4f} "
                  f"{class_metrics['support']:<10}")
            
        print("-" * 80)
        print("\n")


def calculate_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1
) -> Tuple[float, float]:
    tp = ((y_pred == positive_class) & (y_true == positive_class)).sum()
    fp = ((y_pred == positive_class) & (y_true != positive_class)).sum()
    tn = ((y_pred != positive_class) & (y_true != positive_class)).sum()
    fn = ((y_pred != positive_class) & (y_true == positive_class)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return sensitivity, specificity


def compute_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    
    n_classes = cm.shape[0]
    n_samples = cm.sum()
    
    observed_agreement = np.trace(cm) / n_samples
    
    expected_agreement = 0.0
    for i in range(n_classes):
        row_sum = cm[i, :].sum()
        col_sum = cm[:, i].sum()
        expected_agreement += row_sum * col_sum
        
    expected_agreement /= (n_samples * n_samples)
    
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement) \
            if (1 - expected_agreement) > 0 else 0.0
            
    return kappa


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
