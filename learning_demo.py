#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于IPOT的生成模型实现

主要改进：
1. 支持多种目标分布（正态、均匀、混合）
2. 简化绘图逻辑，加快图像显示
3. 减少评估数据量
4. 调整IPOT参数提升性能
"""

import ipot
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from scipy.stats import wasserstein_distance

# 设置全局字体和图表样式
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')

# ===== 超参数配置 =====
class Config:
    # 数据参数
    data_type = 'gaussian'  # 可选 'uniform', 'gaussian', 'mixture'
    data_size = 50000      # 数据集大小
    
    # 模型参数
    z_dim = 1              # 噪声维度
    x_dim = 1              # 生成数据维度
    g_dim = 128            # 生成器隐藏层维度
    batchsize = 512        # 批大小
    
    # 训练参数
    lr = 1e-4              # 学习率
    epochs = 1000          # 训练轮数
    beta = 0.05            # IPOT参数
    num_proximal = 50      # IPOT迭代次数（减少以加速）
    
    # 可视化参数
    plot_interval = 200    # 增加绘图间隔，减少绘图次数

config = Config()

# ===== 数据生成与处理 =====
class DataGenerator:
    """生成各种类型的目标分布数据"""
    
    @staticmethod
    def uniform(low=0, high=2, size=config.data_size):
        return np.random.uniform(low, high, size)
    
    @staticmethod
    def gaussian(mean=0, std=1, size=config.data_size):
        return np.random.normal(mean, std, size)
    
    @staticmethod
    def mixture(size=config.data_size):
        g1 = np.random.normal(-2, 1, size//2)
        g2 = np.random.normal(2, 0.5, size - size//2)
        return np.concatenate([g1, g2])
    
    @classmethod
    def generate_data(cls, data_type):
        if data_type == 'uniform':
            return cls.uniform()
        elif data_type == 'gaussian':
            return cls.gaussian()
        elif data_type == 'mixture':
            return cls.mixture()
        else:
            raise ValueError(f"未知数据类型: {data_type}")

# 生成目标数据和噪声数据
np.random.seed(42)
target_data = DataGenerator.generate_data(config.data_type)
noise_data = np.random.uniform(-1, 1, config.data_size)

# ===== 可视化函数 =====
def plot_comparison(generated, target, iteration, bins=20, show_wdist=True):
    """绘制生成数据与目标数据对比图，简化绘图流程"""
    plt.figure(figsize=(10, 6))
    
    # 绘制目标分布
    if config.data_type == 'uniform':
        plt.plot([target.min(), target.min(), target.max(), target.max()],
                 [0, 1/(target.max()-target.min()), 1/(target.max()-target.min()), 0],
                 color='darkred', lw=3, label='Target Distribution')
    elif config.data_type == 'gaussian':
        x = np.linspace(target.mean() - 2*target.std(), target.mean() + 2*target.std(), 20)
        pdf = np.exp(-0.5*((x-target.mean())/target.std())**2)/(target.std()*np.sqrt(2*np.pi))
        plt.plot(x, pdf, color='darkred', lw=3, label='Target Distribution')
    else:  # mixture
        hist, edges = np.histogram(target, bins=bins, density=True)
        plt.plot(edges[:-1], hist, color='darkred', lw=3, label='Target Distribution')
    
    # 绘制生成数据直方图
    plt.hist(generated, bins=bins, density=True,
             alpha=0.7, color='skyblue', edgecolor='navy',
             label='Generated Data')
    
    if show_wdist:
        w_dist = wasserstein_distance(generated, target)
        plt.title(f'Iteration: {iteration}\nWasserstein Distance: {w_dist:.4f}', fontsize=14)
    else:
        plt.title(f'Iteration: {iteration}', fontsize=14)
        
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

# ===== 数据批处理 =====
def get_batch(data, iteration, batch_size):
    start = (iteration * batch_size) % (len(data) - batch_size)
    return data[start:start + batch_size]

# ===== 模型定义 =====
class Generator(nn.Module):
    """增强版生成器网络（3层MLP）"""
    
    def __init__(self, z_dim, g_dim, x_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, g_dim),
            nn.BatchNorm1d(g_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(g_dim, g_dim//2),
            nn.BatchNorm1d(g_dim//2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(g_dim//2, x_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        return self.net(z)

# ===== 模型初始化 =====
generator = Generator(config.z_dim, config.g_dim, config.x_dim)
optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))

# ===== 训练过程 =====
print(f'Training started for {config.data_type} distribution...')
start_time = time.time()
loss_history = []
w_dist_history = []

for epoch in range(config.epochs):
    y_batch = get_batch(target_data, epoch, config.batchsize)
    z_batch = get_batch(noise_data, epoch, config.batchsize)
    
    z_tensor = torch.FloatTensor(z_batch.reshape(-1, config.z_dim))
    with torch.no_grad():
        x_batch = generator(z_tensor).numpy().flatten()
    
    # ===== IPOT计算 =====
    x_tile = x_batch.reshape(-1, 1)
    y_tile = y_batch.reshape(1, -1)
    deltaC = x_tile - y_tile
    C = deltaC ** 2
    C = C / C.max()
    
    ones = np.ones(config.batchsize)
    P = ipot.ipot_WD(ones, ones, C, beta=config.beta, max_iter=config.num_proximal, return_loss=False)
    
    W = np.sum(P * C)
    update = np.sum(P * deltaC, axis=1)
    
    # ===== 模型更新 =====
    optimizer.zero_grad()
    z_tensor = torch.FloatTensor(z_batch.reshape(-1, config.z_dim))
    x_generated = generator(z_tensor)
    update_tensor = torch.FloatTensor(update.reshape(-1, 1))
    
    loss = torch.sum(update_tensor * x_generated)
    loss.backward()
    optimizer.step()
    
    # 记录损失和距离
    loss_history.append(W)
    current_wdist = wasserstein_distance(x_batch, y_batch)
    w_dist_history.append(current_wdist)
    
    # 定期输出训练信息和可视化
    if epoch % config.plot_interval == 0 or epoch == config.epochs - 1:
        print(f'Epoch: {epoch:4d} | Loss: {W:.4f} | W-Dist: {current_wdist:.4f} | Time: {time.time()-start_time:.2f}s')
        
        with torch.no_grad():
            z_eval = torch.FloatTensor(noise_data[:2000].reshape(-1, config.z_dim))  # 减少样本数
            generated_samples = generator(z_eval).numpy().flatten()
        
        plot_comparison(generated_samples, target_data, epoch, show_wdist=False)  # 关闭WD计算绘图

# ===== 最终结果可视化 =====
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='IPOT Loss')
plt.title('Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(w_dist_history, color='darkgreen', label='Wasserstein Distance')
plt.title('Wasserstein Distance', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

with torch.no_grad():
    z_final = torch.FloatTensor(noise_data.reshape(-1, config.z_dim))
    final_samples = generator(z_final).numpy().flatten()

plot_comparison(final_samples, target_data, 'Final Result')
print("Training completed!")