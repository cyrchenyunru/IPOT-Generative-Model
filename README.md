# IPOT-Generative-Model
A PyTorch implementation of generative models based on the IPOT (Fast Proximal Point Method for Wasserstein Distance) algorithm.

## 项目背景

传统生成模型（如GAN）依赖判别器训练，易出现模式崩溃、训练不稳定等问题。本项目基于**最优传输理论**，采用IPOT算法直接计算生成数据与目标数据的Wasserstein距离，替代判别器实现模型优化，兼具训练稳定性与快速收敛特性。

## 核心特性

1. **多目标分布支持**：一键切换均匀分布、正态分布、混合高斯分布的生成任务
2. **增强型生成器**：3层MLP结构，结合BatchNorm与LeakyReLU，缓解梯度消失
3. **高效IPOT实现**：优化的近端点迭代算法，支持成本矩阵归一化，提升数值稳定性
4. **完整可视化工具**：实时监控训练损失、Wasserstein距离及分布匹配效果
5. **灵活配置**：超参数集中管理，支持批量大小、学习率、迭代次数等参数自定义

## 技术栈

- 深度学习框架：PyTorch 1.0+
- 数值计算：NumPy、SciPy
- 可视化：Matplotlib
- 最优传输：POT库、IPOT算法实现

## 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-username/IPOT-Generative-Model.git
cd IPOT-Generative-Model

# 安装依赖
pip install torch numpy scipy matplotlib pot
```

### 核心参数配置

修改 `learning_demo.py` 中 `Config` 类，支持快速切换实验设置：

```python
class Config:
    data_type = 'gaussian'  # 可选：'uniform'/'gaussian'/'mixture'
    data_size = 50000       # 数据集规模
    batchsize = 512         # 批处理大小
    lr = 1e-4               # 学习率
    epochs = 1000           # 训练轮数
    beta = 0.05             # IPOT算法步长参数
    plot_interval = 200     # 可视化间隔（轮数）
```

### 运行训练

```bash
python learning_demo.py
```

## 项目结构

```
IPOT-Generative-Model/
├── ipot.py                # IPOT算法核心实现（最优传输、 barycenter计算）
├── learning_demo.py       # 主训练脚本（生成模型、训练流程、可视化）
├── barycenter_demo.py     # Wasserstein barycenter计算示例
├── ipot_demo.py           # IPOT算法收敛性验证示例
└── README.md              # 项目说明文档
```

## 核心算法说明

### IPOT算法

IPOT（Fast Proximal Point Method for Wasserstein Distance）是一种高效的最优传输求解算法，通过近端点迭代降低计算复杂度，核心优势：

1. 避免传统EMD算法的高复杂度（O(n³)→O(n²k)，k为迭代次数）
2. 支持批量计算，适配生成模型训练场景
3. 可通过`beta`参数调节收敛速度与稳定性

### 生成模型优化流程

1. 生成器将均匀噪声映射为候选数据
2. 计算生成数据与目标数据的平方欧氏距离矩阵（归一化处理）
3. 调用IPOT算法求解最优传输矩阵，计算Wasserstein距离
4. 基于传输矩阵梯度更新生成器参数，最小化分布差异

## 项目亮点

1. **算法创新**：采用IPOT替代传统GAN判别器，解决训练不稳定问题
2. **工程优化**：成本矩阵归一化、权重初始化策略提升数值稳定性
3. **可视化完善**：实时输出分布对比图、指标曲线，便于结果分析
4. **代码复用性**：模块化设计，支持快速扩展至其他目标分布

## 未来改进方向

1. 扩展至高维数据生成（如图像），引入卷积神经网络（CNN）
2. 融合Sinkhorn算法，进一步降低最优传输计算复杂度
3. 增加注意力机制，提升复杂分布（如混合分布）的细节匹配精度
4. 引入更多定量评估指标（KL散度、MMD等）
