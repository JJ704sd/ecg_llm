import torch
import argparse
import yaml
from pathlib import Path
import sys

from models.ecg_lm import MultiTaskECGLM
from data.dataset import get_dataloader, SyntheticECGDataset
from data.preprocess import ECGPreprocessor
from train.trainer import ECGTrainer, ECGPretrainer
from inference.predictor import ECGInferencer, ECGReportGenerator
from utils.metrics import ECGMetrics


def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict):
    model = MultiTaskECGLM(
        input_channels=config['model']['input_channels'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        conv_channels=config['model']['conv_channels'],
        num_classes=config['data']['num_classes'],
        dropout=config['model']['dropout'],
        max_seq_len=config['model']['max_seq_len'],
    )
    return model


def train_command(args):
    config = load_config(args.config)
    
    print("Creating model...")
    model = create_model(config)
    
    print("Loading data...")
    train_loader = get_dataloader(
        data_dir=config['data']['train_dir'],
        split='train',
        batch_size=config['training']['batch_size'],
        use_synthetic=True,
        synthetic_num_samples=1000,
        num_leads=config['data']['num_leads'],
        sampling_rate=config['data']['sampling_rate'],
        sequence_length=config['data']['sequence_length'],
        num_classes=config['data']['num_classes'],
    )
    
    val_loader = get_dataloader(
        data_dir=config['data']['val_dir'],
        split='val',
        batch_size=config['training']['batch_size'],
        use_synthetic=True,
        synthetic_num_samples=200,
        num_leads=config['data']['num_leads'],
        sampling_rate=config['data']['sampling_rate'],
        sequence_length=config['data']['sequence_length'],
        num_classes=config['data']['num_classes'],
    )
    
    print("Starting training...")
    trainer = ECGTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['inference']['device'],
        checkpoint_dir=config['system']['checkpoint_dir'],
        log_dir=config['system']['log_dir'],
    )
    
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping']['patience'],
    )
    
    print("Training completed!")


def pretrain_command(args):
    config = load_config(args.config)
    
    print("Creating model...")
    model = create_model(config)
    
    print("Loading data for pretraining...")
    train_loader = get_dataloader(
        data_dir=config['data']['train_dir'],
        split='train',
        batch_size=config['training']['batch_size'],
        use_synthetic=True,
        synthetic_num_samples=2000,
        num_leads=config['data']['num_leads'],
        sampling_rate=config['data']['sampling_rate'],
        sequence_length=config['data']['sequence_length'],
        num_classes=config['data']['num_classes'],
    )
    
    print("Starting pretraining...")
    pretrainer = ECGPretrainer(
        model=model,
        train_loader=train_loader,
        device=config['inference']['device'],
        checkpoint_dir=config['system']['checkpoint_dir'],
        log_dir=config['system']['log_dir'],
        masked_ratio=config['model']['pretrain']['masked_ratio'],
    )
    
    pretrainer.pretrain(num_epochs=50)
    
    print("Pretraining completed!")


def eval_command(args):
    config = load_config(args.config)
    
    print("Creating model...")
    model = create_model(config)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Loading test data...")
    test_loader = get_dataloader(
        data_dir=config['data']['test_dir'],
        split='test',
        batch_size=config['training']['batch_size'],
        use_synthetic=True,
        synthetic_num_samples=500,
        num_leads=config['data']['num_leads'],
        sampling_rate=config['data']['sampling_rate'],
        sequence_length=config['data']['sequence_length'],
        num_classes=config['data']['num_classes'],
    )
    
    print("Evaluating...")
    model.eval()
    metrics = ECGMetrics()
    
    device = config['inference']['device']
    model.to(device)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            if isinstance(outputs, dict):
                preds = outputs['classification_logits'].argmax(dim=-1)
                probs = torch.softmax(outputs['classification_logits'], dim=-1)
            else:
                preds = outputs.argmax(dim=-1)
                probs = torch.softmax(outputs, dim=-1)
            
            metrics.update(preds.cpu().numpy(), targets.numpy(), probs.cpu().numpy())
    
    results = metrics.compute()
    metrics.print_summary()
    
    if args.output:
        metrics.save_metrics(args.output)


def infer_command(args):
    config = load_config(args.config)
    
    print("Loading model...")
    model = create_model(config)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    inferencer = ECGInferencer(
        model=model,
        device=config['inference']['device'],
        class_names=config['data']['classes'],
    )
    
    print("Generating sample inference...")
    import numpy as np
    sample_ecg = np.random.randn(12, 5000).astype(np.float32)
    
    result = inferencer.predict(sample_ecg)
    
    print("\n" + "="*50)
    print("Prediction Result:")
    print("="*50)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nProbabilities:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.2%}")
    
    if args.report:
        report_gen = ECGReportGenerator(config['data']['classes'])
        report = report_gen.generate_report(result)
        print("\n" + report)


def main():
    parser = argparse.ArgumentParser(description='ECG Large Model Training and Inference')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--config', type=str, default='configs/base_config.yaml',
                              help='Path to config file')
    parser_train.set_defaults(func=train_command)
    
    parser_pretrain = subparsers.add_parser('pretrain', help='Pretrain the model')
    parser_pretrain.add_argument('--config', type=str, default='configs/base_config.yaml',
                                 help='Path to config file')
    parser_pretrain.set_defaults(func=pretrain_command)
    
    parser_eval = subparsers.add_parser('eval', help='Evaluate the model')
    parser_eval.add_argument('--config', type=str, default='configs/base_config.yaml',
                            help='Path to config file')
    parser_eval.add_argument('--checkpoint', type=str, required=True,
                             help='Path to checkpoint file')
    parser_eval.add_argument('--output', type=str, default=None,
                            help='Path to save evaluation results')
    parser_eval.set_defaults(func=eval_command)
    
    parser_infer = subparsers.add_parser('infer', help='Run inference')
    parser_infer.add_argument('--config', type=str, default='configs/base_config.yaml',
                              help='Path to config file')
    parser_infer.add_argument('--checkpoint', type=str, required=True,
                             help='Path to checkpoint file')
    parser_infer.add_argument('--report', action='store_true',
                             help='Generate diagnostic report')
    parser_infer.set_defaults(func=infer_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    args.func(args)


if __name__ == '__main__':
    main()
