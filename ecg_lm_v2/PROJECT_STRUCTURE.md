# ==============================================================================
# 基于大模型的心电信号处理算法研究 - ECG-LM
# ==============================================================================
# Project: ECG Large Model for Cardiac Signal Processing
# Author: 
# Description: 基于Transformer的心电信号处理大模型
# ==============================================================================

ECG_LM/
├── configs/                    # 配置文件目录
│   ├── base_config.yaml       # 基础配置
│   ├── pretrain_config.yaml  # 预训练配置
│   ├── finetune_config.yaml  # 微调配置
│   └── experiment/           # 实验配置
│       ├── ecg_classify.yaml
│       └── arrhythmia_detect.yaml
│
├── data/                      # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py            # 数据集加载
│   ├── transforms.py         # 数据增强
│   ├── preprocess.py         # 信号预处理
│   └── dataloader.py         # 数据加载器
│
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── backbone/             # 主干网络
│   │   ├── __init__.py
│   │   ├── transformer.py    # Transformer编码器
│   │   ├── conv_module.py    # CNN特征提取
│   │   └── positional_encoding.py
│   ├── pretrain/             # 预训练任务
│   │   ├── __init__.py
│   │   ├── masked_modeling.py  # 掩码建模
│   │   └── contrastive.py      # 对比学习
│   ├── heads/                # 任务头
│   │   ├── __init__.py
│   │   ├── classifier.py     # 分类头
│   │   ├── detector.py       # 检测头
│   │   └── report_gen.py     # 报告生成头
│   ├── ecg_lm.py             # 完整模型
│   └── loss.py               # 损失函数
│
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── metrics.py            # 评估指标
│   ├── logger.py             # 日志工具
│   ├── visualizer.py         # 可视化
│   └── helpers.py            # 辅助函数
│
├── train/                    # 训练脚本
│   ├── __init__.py
│   ├── trainer.py            # 训练器
│   ├── pretrain.py           # 预训练入口
│   ├── finetune.py           # 微调入口
│   └── eval.py               # 评估入口
│
├── inference/                # 推理模块
│   ├── __init__.py
│   ├── predictor.py          # 预测器
│   ├── api.py                # API服务
│   └── cli.py                # 命令行接口
│
├── scripts/                  # 辅助脚本
│   ├── download_data.py      # 数据下载
│   ├── convert_format.py     # 格式转换
│   ├── analyze_data.py       # 数据分析
│   └── export_model.py       # 模型导出
│
├── logs/                     # 日志目录
├── checkpoints/             # 模型权重目录
├── outputs/                  # 输出结果目录
├── requirements.txt          # 依赖
├── setup.py                  # 安装脚本
└── README.md                 # 项目说明
