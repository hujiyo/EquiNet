# EquiNet

## 项目简介

EquiNet 是一个基于 Transformer 架构的股票未来走势概率预测模型。项目旨在通过深度学习方法，利用历史股票及大盘数据，预测未来3天内股票上涨、下跌或平稳的概率。支持灵活的模型参数配置和高效的GPU训练，适合金融量化、学术研究等场景。

## 软件架构

- Python 3.8+
- PyTorch
- pandas、numpy
- tqdm

核心模块包括数据预处理、样本生成、模型定义（Transformer）、训练与评估。

## 安装教程

1. 克隆本仓库
   ```bash
   git clone https://gitee.com/yourname/EquiNet.git
   cd EquiNet
   ```
2. 安装依赖
   ```bash
   pip install torch pandas numpy tqdm
   ```
3. 准备数据
   - 将319个`.xlsx`股票数据文件放入`./data`目录下，每个文件包含420天的股票及大盘信息。

## 使用说明

1. 运行训练脚本
   ```bash
   python src/train.py
   ```
2. 按提示输入模型参数（如模型维度、训练轮数等），程序会说明参数对模型效果和参数量的影响。
3. 训练过程中会显示进度、损失、学习率及每轮评估准确率。
4. 训练完成后，模型权重保存在`./out/EquiNet{d_model}.pth`。

## 功能亮点

- **灵活参数配置**：支持自定义模型大小、训练轮数等，适应不同硬件和需求。
- **完全随机训练样本**：每轮训练均从全体股票数据中随机采样，提升泛化能力。
- **归一化处理**：不同股票数据归一化，模型更关注内在规律。
- **GPU加速**：自动检测CUDA，充分利用GPU资源。
- **训练过程评估**：每轮训练后自动评估模型当前预测准确率。

## 参与贡献

1. Fork 本仓库
2. 新建分支（如 `feat_xxx`）
3. 提交代码
4. 新建 Pull Request

## 其他

- 支持多语言文档（如 Readme_en.md, Readme_zh.md）
- 更多开源项目可参考 [Gitee Explore](https://gitee.com/explore)
