# ViT手写体分类器

## 项目简介

本项目基于 [phpy]实现，利用PHP调用Python编写的一个视觉Transformer模型（Vision Transformer，ViT），用于对手写体数字进行分类。该项目主要使用了ViT模型对经典的MNIST手写体数字数据集进行分类任务。项目结合了PHP和Python的优势，利用PHP代码直接调用Python的深度学习框架PyTorch，实现复杂的神经网络模型训练和推理。

## 主要功能

- **ViT模型**: 使用Vision Transformer模型对手写体数字进行分类。
- **数据处理**: 使用Python的torchvision库下载和处理MNIST数据集。
- **训练和推理**: 使用PHP代码调用PyTorch的训练流程，并进行模型推理。
- **模型保存**: 支持保存和加载训练好的模型参数。

## 项目结构

- `train.php`: 主文件,训练Vision Transformer模型。
- `vit.php`: 定义Vision Transformer模型。
- `dataset.php`: 数据集文件，处理MNIST手写体数据集。
- `model.pth`: 保存训练好的模型参数文件（如果存在）。
- `mnist/`: 存放MNIST数据集的目录。
- `python/`: 存放该项目对于python代码的写法。

## 环境依赖

- PHP 8.1 或更高版本
- Python 3.8 或更高版本
- [phpy](https://github.com/swoole/phpy) 扩展
- PyTorch
- torchvision
- matplotlib（用于可视化）

## 安装与使用

### 1. 安装phpy扩展

请按照 [phpy 的官方文档](https://github.com/swoole/phpy) 安装并配置phpy扩展，以使PHP能够调用Python。

### 2. 安装Python依赖

在命令行中执行以下命令，安装必要的Python库：

```bash
pip install torch torchvision matplotlib
```

### 3. 运行项目

在项目根目录下，通过以下命令运行PHP文件以启动模型训练和测试：

```bash
php vit.php
```

你还可以运行 `dataset.php` 来查看MNIST数据集的加载情况：

```bash
php dataset.php
```

### 4. 保存和加载模型

训练过程中，模型参数会自动保存在`model.pth`文件中。下次运行时，程序会尝试从该文件加载模型参数。

## 示例输出

训练过程中，程序会输出每1000次迭代的损失值，并在每次保存模型参数时给出提示。

```
epoch:0 iter:0,loss:2.303
epoch:0 iter:1000,loss:0.278
...
```

## 贡献

欢迎贡献代码、报告问题或提出改进建议。你可以通过GitHub Issue或Pull Request与我们联系。

## 许可证

本项目基于MIT许可证发布，详情请参阅[LICENSE](./LICENSE)文件。