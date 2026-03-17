import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class ECGInferencer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        if class_names is None:
            self.class_names = [
                "Normal", "AFib", "AFL", "SVT", "VT", 
                "PVC", "PAC", "LBBB", "RBBB", "Brady", "Tachy", "MI"
            ]
        else:
            self.class_names = class_names
            
    def predict(
        self, 
        ecg_signal: np.ndarray,
        return_probs: bool = True,
    ) -> Dict:
        if ecg_signal.ndim == 2:
            ecg_signal = ecg_signal[np.newaxis, :, :]
            
        ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(ecg_tensor)
            
            if isinstance(outputs, dict):
                logits = outputs['classification_logits']
            else:
                logits = outputs
                
            probs = F.softmax(logits, dim=-1)
            
            pred_id = probs.argmax(dim=-1).item()
            pred_class = self.class_names[pred_id]
            confidence = probs[0, pred_id].item()
            
        result = {
            'prediction': pred_class,
            'class_id': pred_id,
            'confidence': confidence,
        }
        
        if return_probs:
            result['probabilities'] = {
                name: probs[0, i].item() 
                for i, name in enumerate(self.class_names)
            }
            
        return result
    
    def predict_batch(
        self,
        ecg_signals: np.ndarray,
    ) -> List[Dict]:
        if ecg_signals.ndim == 2:
            ecg_signals = ecg_signals[np.newaxis, :, :]
            
        batch_size = ecg_signals.shape[0]
        
        ecg_tensor = torch.tensor(ecg_signals, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(ecg_tensor)
            
            if isinstance(outputs, dict):
                logits = outputs['classification_logits']
            else:
                logits = outputs
                
            probs = F.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1)
            
        results = []
        for i in range(batch_size):
            result = {
                'prediction': self.class_names[pred_ids[i].item()],
                'class_id': pred_ids[i].item(),
                'confidence': probs[i, pred_ids[i]].item(),
                'probabilities': {
                    name: probs[i, j].item() 
                    for j, name in enumerate(self.class_names)
                }
            }
            results.append(result)
            
        return results
    
    def visualize_prediction(
        self,
        ecg_signal: np.ndarray,
        result: Dict,
        save_path: Optional[str] = None,
    ):
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        for lead_idx in range(min(12, ecg_signal.shape[0])):
            ax = axes[lead_idx]
            time_axis = np.arange(ecg_signal.shape[1]) / 500
            ax.plot(time_axis, ecg_signal[lead_idx], 'b-', linewidth=0.5)
            ax.set_title(f"Lead {lead_idx + 1}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            
        for idx in range(12, len(axes)):
            axes[idx].axis('off')
            
        pred_text = f"Prediction: {result['prediction']}\nConfidence: {result['confidence']:.2%}"
        axes[11].text(0.5, 0.5, pred_text, ha='center', va='center', 
                     fontsize=14, transform=axes[11].transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle("ECG Prediction Result", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            
        plt.close()
        
    def visualize_attention(
        self,
        ecg_signal: np.ndarray,
        attention_weights: List[torch.Tensor],
        save_path: Optional[str] = None,
    ):
        num_layers = len(attention_weights)
        
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(20, 8))
        axes = axes.flatten()
        
        for layer_idx, attn in enumerate(attention_weights[:6]):
            if isinstance(attn, list):
                attn = attn[0]
                
            attn = attn[0].cpu().numpy()
            
            ax = axes[layer_idx]
            im = ax.imshow(attn[:50, :50], cmap='viridis', aspect='auto')
            ax.set_title(f"Layer {layer_idx + 1}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            plt.colorbar(im, ax=ax)
            
        for idx in range(len(attention_weights[:6]), len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle("Attention Weights Visualization", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Attention figure saved to {save_path}")
            
        plt.close()


class ECGReportGenerator:
    def __init__(self, class_names: Optional[List[str]] = None):
        if class_names is None:
            self.class_names = [
                "Normal", "AFib", "AFL", "SVT", "VT", 
                "PVC", "PAC", "LBBB", "RBBB", "Brady", "Tachy", "MI"
            ]
        else:
            self.class_names = class_names
            
    def generate_report(
        self,
        prediction: Dict,
        patient_info: Optional[Dict] = None,
    ) -> str:
        lines = []
        
        lines.append("=" * 60)
        lines.append("                    心电图分析报告")
        lines.append("                  ECG Analysis Report")
        lines.append("=" * 60)
        lines.append("")
        
        if patient_info:
            lines.append("【患者信息】")
            if 'name' in patient_info:
                lines.append(f"  姓名: {patient_info['name']}")
            if 'age' in patient_info:
                lines.append(f"  年龄: {patient_info['age']}")
            if 'gender' in patient_info:
                lines.append(f"  性别: {patient_info['gender']}")
            lines.append("")
            
        lines.append("【诊断结果】")
        lines.append(f"  主要诊断: {prediction['prediction']}")
        lines.append(f"  置信度: {prediction['confidence']:.2%}")
        lines.append("")
        
        probs = prediction.get('probabilities', {})
        if probs:
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            lines.append("【分类概率】")
            for class_name, prob in sorted_probs[:5]:
                bar = "█" * int(prob * 20)
                lines.append(f"  {class_name:10s}: {prob:6.2%} {bar}")
            lines.append("")
            
        lines.append("【临床建议】")
        lines.append(self._get_clinical_suggestion(prediction['prediction']))
        lines.append("")
        
        lines.append("=" * 60)
        lines.append("注: 本报告由AI辅助生成，仅供参考")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _get_clinical_suggestion(self, diagnosis: str) -> str:
        suggestions = {
            "Normal": "心电图检查结果未见明显异常。建议保持健康的生活方式，定期体检。",
            "AFib": "检测到房颤，建议及时就医进行进一步评估和治疗。房颤可能增加中风风险，需密切关注。",
            "AFL": "检测到心房扑动，建议进行心脏电生理检查和相应治疗。",
            "SVT": "检测到室上性心动过速，如有症状建议就医评估。",
            "VT": "检测到室性心动过速，这可能是一种严重的心律失常，建议立即就医！",
            "PVC": "检测到室性早搏，如频繁出现建议进行进一步检查。",
            "PAC": "检测到房性早搏，如频繁出现建议进行进一步检查。",
            "LBBB": "检测到左束支传导阻滞，建议结合临床症状进行评估。",
            "RBBB": "检测到右束支传导阻滞，可见于正常人群，如无症状可定期复查。",
            "Brady": "检测到心动过缓，如有头晕、乏力等症状建议就医。",
            "Tachy": "检测到心动过速，建议复查并评估原因。",
            "MI": "检测到心肌梗死可能，请立即就医！",
        }
        
        return suggestions.get(diagnosis, "建议咨询专科医生进行进一步评估。")


def load_model_for_inference(
    checkpoint_path: str,
    model_class,
    device: str = "cuda",
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def batch_inference(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
) -> List[Dict]:
    inferencer = ECGInferencer(model, device=device)
    
    results = []
    
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            ecg_signals = batch[0].numpy()
        else:
            ecg_signals = batch.numpy()
            
        batch_results = inferencer.predict_batch(ecg_signals)
        results.extend(batch_results)
        
    return results
