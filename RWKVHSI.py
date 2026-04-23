import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, reduce, repeat




# RWKV时间混合模块（高效序列建模）
class TimeMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.time_decay = nn.Parameter(torch.randn(dim))

        # 门控机制
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        # 自适应归一化
        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x, state=None):
        B, T, C = x.shape

        # 门控计算
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        # WKV计算
        wkv, state_out = self.wkv_computation(k, v, state)

        # 输出融合
        r = torch.sigmoid(r)
        out = r * wkv
        out = self.output(out)
        out = self.ln_out(out)
        return out, state_out

    def wkv_computation(self, k, v, state=None):
        """高效WKV计算（线性复杂度）"""
        B, T, C = k.shape

        # 初始化状态
        if state is None:
            state = torch.zeros(B, C, device=k.device)

        # 创建时间衰减矩阵
        decay = -torch.exp(self.time_decay)

        # 累积计算
        wkv = torch.zeros(B, T, C, device=k.device)
        accum = state.clone()

        for t in range(T):
            curr_k = k[:, t]
            curr_v = v[:, t]

            # 使用切片赋值
            wkv[:, t, :] = accum + curr_v

            accum = accum * torch.exp(decay) + curr_k * curr_v

        return wkv, accum


# 通道混合模块（特征变换）
class ChannelMix(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.gate = nn.Linear(dim, dim * expansion)

    def forward(self, x):
        x = self.ln(x)
        gate = torch.sigmoid(self.gate(x))
        x = self.fc1(x) * gate
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# RWKV基础块（光谱/空间）
class RWKVBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timemix = TimeMix(dim)
        self.channelmix = (dim)

    def forward(self, x, state=None):
        residual = x
        x, state_out = self.timemix(x, state)
        x = residual + x

        residual = x
        x = self.channelmix(x)
        x = residual + x

        return x, state_out


# 光谱流处理模块（带状态管理）- 修复版
class SpectralStream(nn.Module):
    def __init__(self, in_dim, embed_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.embed = nn.Conv3d(embed_dim, embed_dim, (1, 1, 1), stride=1, padding=0)  # 使用1x1x1卷积核
        self.blocks = nn.ModuleList([RWKVBlock(embed_dim) for _ in range(depth)])
        self.ln = nn.LayerNorm(embed_dim)
        # 添加线性层处理光谱维度
        self.spectral_proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x, state=None):
        # 输入: [B, C, H, W]
        B, C, H, W = x.shape

        # 先投影光谱维度
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.spectral_proj(x)  # [B, H, W, D]
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W]

        # 添加维度以适应3D卷积
        x = x.unsqueeze(2)  # [B, D, 1, H, W]

        # 应用3D卷积
        # print(x.shape)
        x = self.embed(x)  # [B, D, 1, H, W]
        x = x.squeeze(2)  # [B, D, H, W]

        # 重排为序列格式 [B, H, W, D] -> [B*H, W, D]
        x = x.permute(0, 2, 3, 1).reshape(B * H, W, -1)

        # 初始化状态
        if state is None:
            state = [None] * self.depth

        new_state = []
        # 光谱特征提取
        for i, block in enumerate(self.blocks):
            x, s = block(x, state[i])
            new_state.append(s)

        x = self.ln(x)
        # 重排回空间格式 [B*H, W, D] -> [B, D, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x, new_state


# 空间流处理模块（带状态管理）
class SpatialStream(nn.Module):
    def __init__(self, in_dim, embed_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.conv = nn.Conv2d(in_dim, embed_dim, 3, padding=1)
        self.blocks = nn.ModuleList([RWKVBlock(embed_dim) for _ in range(depth)])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, state=None):
        # 输入: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).reshape(B * H, W, -1)

        # 初始化状态
        if state is None:
            state = [None] * self.depth

        new_state = []
        # 空间特征提取
        for i, block in enumerate(self.blocks):
            x, s = block(x, state[i])
            new_state.append(s)

        x = self.ln(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x, new_state


# 自适应特征融合模块（优化版）
class AdaptiveFusion(nn.Module):
    def __init__(self, in_dim, spectral_dim, spatial_dim):
        super().__init__()
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(spectral_dim, in_dim, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(spatial_dim, in_dim, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1),
            nn.BatchNorm2d(in_dim),
            nn.GELU()
        )

    def forward(self, spectral_feat, spatial_feat):
        spectral_weight = self.spectral_gate(spectral_feat)
        spatial_weight = self.spatial_gate(spatial_feat)

        # 门控融合
        gated_spectral = spectral_feat * spectral_weight
        gated_spatial = spatial_feat * spatial_weight

        # 拼接融合
        fused = torch.cat([gated_spectral, gated_spatial], dim=1)
        return self.fusion_conv(fused)


# 混合层级模块
class HybridLevel(nn.Module):
    def __init__(self, in_dim, embed_dim, depth=1):
        super().__init__()
        # 光谱流
        self.spectral_stream = SpectralStream(in_dim, embed_dim, depth)

        # 空间流
        self.spatial_stream = SpatialStream(in_dim, embed_dim, depth)

        # 特征融合
        self.fusion = AdaptiveFusion(embed_dim, embed_dim, embed_dim)

        # 残差连接
        if in_dim != embed_dim:
            self.residual = nn.Conv2d(in_dim, embed_dim, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, spectral_state=None, spatial_state=None):
        residual = self.residual(x)

        # 光谱特征提取
        spectral_feat, new_spectral_state = self.spectral_stream(x, spectral_state)

        # 空间特征提取
        spatial_feat, new_spatial_state = self.spatial_stream(x, spatial_state)

        # 特征融合
        fused = self.fusion(spectral_feat, spatial_feat)

        return fused + residual, new_spectral_state, new_spatial_state


# RWKVHSI主模型（优化版）
class RWKVHSI(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=64, reduced_bands=30, depth=1):
        super().__init__()
        self.depth = depth

        # PCA降维
        #self.pca = PCAReduction(in_channels, reduced_bands)
        self.reduced_bands = reduced_bands

        # 初始卷积
        self.init_conv =  nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            #nn.GELU()
            nn.ReLU()
        )

        # 多层级混合处理
        self.levels = nn.ModuleList()
        for i in range(depth):
            self.levels.append(HybridLevel(embed_dim, embed_dim, depth=1))#3



        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        # 特征存储
        self.feature_maps = {}
        self.states = [None] * depth * 2  # 存储光谱和空间状态

    # def fit_pca(self, data_loader):
    #     """使用训练数据拟合PCA"""
    #     self.pca.fit(data_loader)

    def forward(self, x, visualize=False, ablation=None):
        #print('X:',x.shape)
        x = x.squeeze()
        """ablation: None, 'spectral', 'spatial'"""
        # 存储中间特征
        if visualize:
            self.feature_maps = {}


        # 初始卷积
        x = self.init_conv(x)
        if visualize:
            self.feature_maps['init_conv'] = x.detach().cpu()

        # 初始化状态
        spectral_state = [None] * self.depth
        spatial_state = [None] * self.depth

        # 多层级处理
        for i, level in enumerate(self.levels):
            # 消融实验处理
            if ablation == 'spectral':
                # 仅使用光谱流
                x, spectral_state, _ = level(x, spectral_state, None)
            elif ablation == 'spatial':
                # 仅使用空间流
                x, _, spatial_state = level(x, None, spatial_state)
            else:
                # 正常处理
                x, spectral_state, spatial_state = level(x, spectral_state, spatial_state)

            # 存储状态
            self.states[i * 2] = spectral_state
            self.states[i * 2 + 1] = spatial_state

            if visualize:
                self.feature_maps[f'level_{i}'] = x.detach().cpu()


        # 分类
        x = self.classifier(x)
        return x

    def get_feature_maps(self):
        return self.feature_maps

    def get_states(self):
        return self.states


# 消融实验封装
class AblationModel(nn.Module):
    def __init__(self, base_model, ablation_type='both'):
        """
        ablation_type:
            'both' - 同时使用空间和光谱流
            'spatial' - 仅使用空间流
            'spectral' - 仅使用光谱流
        """
        super().__init__()
        self.base_model = base_model
        self.ablation_type = ablation_type

    def fit_pca(self, data_loader):
        self.base_model.fit_pca(data_loader)

    def forward(self, x, visualize=False):
        return self.base_model(x, visualize, ablation=self.ablation_type)


# 测试和可视化代码
def test_model():
    # 模拟HSI数据
    B, C, H, W = 8, 200, 15, 15
    num_classes = 16

    # 创建数据集
    dataset = TensorDataset(
        torch.randn(B, C, H, W),
        torch.randint(0, num_classes, (B,))
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 创建模型
    model = RWKVHSI(in_channels=C, num_classes=num_classes, embed_dim=64, reduced_bands=30)

    # 拟合PCA
    model.fit_pca(dataloader)

    # 测试完整模型
    x = torch.randn(B, C, H, W)
    y = model(x, visualize=True)
    print(f"完整模型输出尺寸: {y.shape}")

    # 可视化特征
    visualize_features(model, x)

    # 消融实验
    ablation_results = {}
    for ablation_type in ['both', 'spectral', 'spatial']:
        ablation_model = AblationModel(model, ablation_type)
        y = ablation_model(x)
        ablation_results[ablation_type] = y
        print(f"消融实验 [{ablation_type}] 输出尺寸: {y.shape}")

    # 运行消融实验分析
    run_ablation_experiment(model, dataloader)

    return model


def visualize_features(model, input_tensor):
    model.eval()
    with torch.no_grad():
        # 前向传播获取特征
        _ = model(input_tensor, visualize=True)
        features = model.get_feature_maps()

        # 创建可视化
        plt.figure(figsize=(15, 10))
        titles = ['PCA Output', 'Initial Conv'] + \
                 [f'Level {i}' for i in range(model.depth)] + \
                 ['Pyramid Level 1', 'Pyramid Level 2']

        for i, (name, feat) in enumerate(features.items()):
            if i >= len(titles):
                break

            # 取第一个样本的第一个通道
            sample_idx = 0
            channel_idx = 0

            plt.subplot(3, 4, i + 1)
            if feat.dim() == 4:
                plt.imshow(feat[sample_idx, channel_idx].numpy(), cmap='viridis')
            else:
                plt.imshow(feat[sample_idx].numpy(), cmap='viridis')
            plt.colorbar()
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('feature_visualization.png', dpi=300)
        plt.show()

        # 特征重要性分析
        analyze_feature_importance(features, input_tensor)


def analyze_feature_importance(features, input_tensor):
    variances = {}
    for name, feat in features.items():
        feat_flat = feat.reshape(feat.size(0), -1)
        variances[name] = torch.var(feat_flat, dim=1).mean().item()

    # 绘制重要性条形图
    plt.figure(figsize=(12, 6))
    plt.bar(variances.keys(), variances.values(), color='skyblue')
    plt.ylabel('Feature Variance (Importance)')
    plt.title('Feature Importance Analysis')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数值标签
    for i, v in enumerate(variances.values()):
        plt.text(i, v + 0.001, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()


def run_ablation_experiment(model, dataloader, num_epochs=3):
    """运行消融实验并分析性能"""
    results = {}

    # 测试三种配置
    for ablation_type in ['both', 'spectral', 'spatial']:
        # 创建消融模型
        ablation_model = AblationModel(model, ablation_type=ablation_type)

        # 测试性能
        total_correct = 0
        total_samples = 0

        for data, labels in dataloader:
            with torch.no_grad():
                outputs = ablation_model(data)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        results[ablation_type] = accuracy
        print(f"Ablation [{ablation_type}] Accuracy: {accuracy:.4f}")

    # 可视化消融结果
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title('Ablation Study Results')
    plt.xlabel('Model Configuration')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('ablation_results.png', dpi=300)
    plt.close()

    return results


# 执行测试
if __name__ == "__main__":
    model = test_model()