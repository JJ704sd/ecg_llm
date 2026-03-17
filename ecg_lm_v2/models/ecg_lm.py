import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class ECGClassifierHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_classes: int = 12,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ECGDetectorHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_detection_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_detection_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.detector(x)


class ECGSegmentationHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_segments: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.segmentation = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_segments)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmentation(x)


class ECGReportGenerator(nn.Module):
    def __init__(
        self,
        d_model: int,
        vocab_size: int = 1000,
        max_length: int = 100,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(
        self, 
        encoder_output: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = encoder_output.size(0)
        
        if target_ids is not None:
            target_emb = self.embedding(target_ids)
            seq_len = target_ids.size(1)
        else:
            target_emb = torch.zeros(batch_size, 1, self.d_model).to(encoder_output.device)
            seq_len = 1
            
        positions = torch.arange(seq_len).unsqueeze(0).to(encoder_output.device)
        pos_emb = self.position_embedding(positions)
        
        x = target_emb + pos_emb
        
        for block in self.decoder_blocks:
            x = block(x, encoder_output)
            
        logits = self.output_projection(x)
        
        return logits
    
    def generate(
        self,
        encoder_output: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        if max_length is None:
            max_length = self.max_length
            
        batch_size = encoder_output.size(0)
        
        generated = torch.zeros(batch_size, 1).long().to(encoder_output.device)
        
        for _ in range(max_length):
            logits = self.forward(encoder_output, generated)
            
            logits = logits[:, -1, :] / temperature
            
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices.gather(1, torch.multinomial(probs, 1))
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == 0).all():
                break
                
        return generated


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, encoder_output, encoder_output, attn_mask=src_mask)
        x = residual + self.dropout(x)
        
        residual = x
        x = self.norm3(x)
        x = residual + self.feed_forward(x)
        
        return x


class MultiTaskECGLM(nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 12,
        d_ff: int = 1024,
        conv_channels: list = [64, 128, 256],
        num_classes: int = 12,
        num_detection_classes: int = 5,
        num_segment_classes: int = 4,
        vocab_size: int = 1000,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        from .backbone.transformer import ECGTransformerEncoder
        
        self.encoder = ECGTransformerEncoder(
            input_channels=input_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            conv_channels=conv_channels,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        
        self.classifier_head = ECGClassifierHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        self.detector_head = ECGDetectorHead(
            d_model=d_model,
            num_detection_classes=num_detection_classes,
            dropout=dropout,
        )
        
        self.segmentation_head = ECGSegmentationHead(
            d_model=d_model,
            num_segments=num_segment_classes,
            dropout=dropout,
        )
        
        self.report_generator = ECGReportGenerator(
            d_model=d_model,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        
        if task_weights is None:
            task_weights = {
                'classification': 1.0,
                'detection': 0.5,
                'segmentation': 0.3,
                'report': 0.3
            }
        self.task_weights = task_weights
        
    def forward(
        self,
        x: torch.Tensor,
        return_all_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoder_output, attention_weights = self.encoder(x)
        
        pooled_output = encoder_output.mean(dim=1)
        
        classification_logits = self.classifier_head(pooled_output)
        
        detection_logits = self.detector_head(pooled_output)
        
        segmentation_logits = self.segmentation_head(pooled_output)
        
        outputs = {
            'classification_logits': classification_logits,
            'detection_logits': detection_logits,
            'segmentation_logits': segmentation_logits,
            'encoder_output': encoder_output,
            'attention_weights': attention_weights,
        }
        
        if return_all_features:
            outputs['pooled_output'] = pooled_output
            
        return outputs
    
    def generate_report(
        self,
        x: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_output, _ = self.encoder(x)
        
        report_logits = self.report_generator(encoder_output, target_ids)
        
        return report_logits
    
    def predict(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            classification_probs = F.softmax(outputs['classification_logits'], dim=-1)
            detection_probs = F.softmax(outputs['detection_logits'], dim=-1)
            segmentation_probs = F.softmax(outputs['segmentation_logits'], dim=-1)
            
            predictions = {
                'class_ids': classification_probs.argmax(dim=-1),
                'class_probs': classification_probs,
                'detection_ids': detection_probs.argmax(dim=-1),
                'detection_probs': detection_probs,
                'segmentation_ids': segmentation_probs.argmax(dim=-1),
                'segmentation_probs': segmentation_probs,
            }
            
        return predictions


class ECGLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        class_weights: Optional[torch.Tensor] = None,
        task_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.detection_loss = nn.CrossEntropyLoss()
        
        self.segmentation_loss = nn.CrossEntropyLoss()
        
        self.report_loss = nn.CrossEntropyLoss(ignore_index=0)
        
        if task_weights is None:
            task_weights = {
                'classification': 1.0,
                'detection': 0.5,
                'segmentation': 0.3,
                'report': 0.3
            }
        self.task_weights = task_weights
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        classification_loss = self.classification_loss(
            outputs['classification_logits'],
            labels['class_labels']
        )
        
        detection_loss = self.detection_loss(
            outputs['detection_logits'],
            labels['detection_labels']
        )
        
        segmentation_loss = self.segmentation_loss(
            outputs['segmentation_logits'],
            labels['segmentation_labels']
        )
        
        total_loss = (
            self.task_weights['classification'] * classification_loss +
            self.task_weights['detection'] * detection_loss +
            self.task_weights['segmentation'] * segmentation_loss
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'detection_loss': detection_loss,
            'segmentation_loss': segmentation_loss,
        }
        
        if 'report_logits' in outputs and 'report_labels' in labels:
            report_loss = self.report_loss(
                outputs['report_logits'].view(-1, outputs['report_logits'].size(-1)),
                labels['report_labels'].view(-1)
            )
            total_loss = total_loss + self.task_weights['report'] * report_loss
            loss_dict['report_loss'] = report_loss
            loss_dict['total_loss'] = total_loss
            
        return loss_dict


class ECGDiagnosisReport:
    CLASS_NAMES = [
        "Normal", "AFib", "AFL", "SVT", "VT", 
        "PVC", "PAC", "LBBB", "RBBB", "Brady", "Tachy", "MI"
    ]
    
    CLINICAL_SUGGESTIONS = {
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
    
    def __init__(self, language: str = "zh"):
        self.language = language
        
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
            if 'id' in patient_info:
                lines.append(f"  ID: {patient_info['id']}")
            lines.append("")
            
        lines.append("【诊断结果】")
        pred_class = prediction.get('prediction', 'Unknown')
        confidence = prediction.get('confidence', 0.0)
        lines.append(f"  主要诊断: {pred_class}")
        lines.append(f"  置信度: {confidence:.2%}")
        lines.append("")
        
        probs = prediction.get('probabilities', {})
        if probs:
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            lines.append("【分类概率】")
            for class_name, prob in sorted_probs[:5]:
                bar = "█" * int(prob * 20)
                lines.append(f"  {class_name:10s}: {prob:6.2%} {bar}")
            lines.append("")
            
        if pred_class in self.CLINICAL_SUGGESTIONS:
            lines.append("【临床建议】")
            lines.append(self.CLINICAL_SUGGESTIONS[pred_class])
            lines.append("")
        
        detection = prediction.get('detection', {})
        if detection:
            lines.append("【异常检测】")
            det_class = detection.get('prediction', 'None')
            det_conf = detection.get('confidence', 0.0)
            lines.append(f"  检测结果: {det_class}")
            lines.append(f"  置信度: {det_conf:.2%}")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("注: 本报告由AI辅助生成，仅供参考")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def generate_structured_data(
        self,
        prediction: Dict,
    ) -> Dict:
        pred_class = prediction.get('prediction', 'Unknown')
        
        return {
            'diagnosis': pred_class,
            'confidence': float(prediction.get('confidence', 0.0)),
            'class_id': int(prediction.get('class_id', -1)),
            'suggestion': self.CLINICAL_SUGGESTIONS.get(pred_class, ""),
            'all_probabilities': prediction.get('probabilities', {}),
        }


def create_ecg_model(
    model_type: str = "multi_task",
    **kwargs
) -> nn.Module:
    if model_type == "multi_task":
        return MultiTaskECGLM(**kwargs)
    elif model_type == "classifier":
        from .backbone.transformer import ECGTransformerEncoder
        encoder = ECGTransformerEncoder(**kwargs)
        return nn.Sequential(
            encoder,
            nn.Linear(kwargs.get('d_model', 256), kwargs.get('num_classes', 12))
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
