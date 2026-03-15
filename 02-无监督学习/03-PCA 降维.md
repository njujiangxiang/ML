# 3. 主成分分析 (PCA): 降维神器

## 🎯 学习目标
- 理解降维的必要性和目的
- 掌握 PCA 的核心思想：方差最大化
- 学会选择主成分数量
- 理解 PCA 的应用场景和局限性

---

## 一、为什么要降维？

### 问题：高维数据的挑战

```
维度灾难：
- 特征太多 → 数据稀疏
- 计算成本高
- 容易过拟合
- 难以可视化

例子：
100 个特征 → 100 维空间
人类只能理解 2D/3D
如何可视化？如何理解？
```

### 降维的好处

```
┌─────────────────────────────────────────────────────────────┐
│                    降维的好处                                │
├─────────────────────────────────────────────────────────────┤
│ ✓ 减少计算量                                                │
│   100 维 → 2 维，计算快 50 倍                                 │
├─────────────────────────────────────────────────────────────┤
│ ✓ 去除噪声和冗余                                            │
│   相关特征合并，保留关键信息                                │
├─────────────────────────────────────────────────────────────┤
│ ✓ 可视化                                                    │
│   高维数据 → 2D/3D 图                                        │
├─────────────────────────────────────────────────────────────┤
│ ✓ 防止过拟合                                                │
│   减少特征，简化模型                                        │
├─────────────────────────────────────────────────────────────┤
│ ✓ 特征提取                                                  │
│   发现数据的潜在结构                                        │
└─────────────────────────────────────────────────────────────┘
```

### 降维方法分类

```
降维方法
│
├── 线性方法
│   ├── PCA (主成分分析) ← 本章
│   ├── LDA (线性判别分析)
│   └── MDS (多维缩放)
│
└── 非线性方法
    ├── t-SNE ← 下一章
    ├── UMAP
    └── 自编码器
```

---

## 二、PCA 的核心思想

### 直觉理解

```
PCA = 找到数据变化最大的方向

2D → 1D:
  y
  ↑
  │  ● ● ● ●
  │ ● ● ● ●
  │● ● ● ●
  └────────────→ x
  
  数据主要沿对角线方向变化
  
  新坐标轴：
  y
  ↑    ╱ PC1 (主成分 1)
  │  ╱●●●●
  │╱ ●●●●
  │●●●●
  └────────────→ x
  
  投影到 PC1，从 2D 变 1D
  保留最多信息！
```

### 什么是主成分？

```
主成分 (Principal Component):
- 新的坐标轴方向
- 互相垂直 (正交)
- 按方差大小排序

PC1: 方差最大的方向 (第一主成分)
PC2: 与 PC1 垂直，方差次大 (第二主成分)
PC3: 与 PC1、PC2 垂直，方差第三大
...

可视化：

     y
     ↑
     │   ╱ PC2 (方差小)
     │ ╱│
     │╱ │
  ───●───────→ PC1 (方差大)
   ╱│
 ╱  │
```

### 方差 = 信息

```
为什么最大化方差？

方差大 → 数据分散 → 信息多
方差小 → 数据集中 → 信息少

例子：
高方差：[1, 5, 9, 13, 17]  ← 能区分不同样本
低方差：[9, 10, 9, 10, 9]  ← 区分度小

PCA 目标：
找到新坐标轴，让投影后的方差最大
```

---

## 三、PCA 算法原理

### 算法步骤

```
PCA 计算步骤：

1. 数据中心化
   每个特征减去均值
   
2. 计算协方差矩阵
   衡量特征间的相关性
   
3. 计算特征值和特征向量
   特征向量 = 主成分方向
   特征值 = 方差大小
   
4. 排序并选择主成分
   按特征值从大到小排序
   选择前 K 个主成分
   
5. 投影到新空间
   原始数据 × 特征向量矩阵
```

### 数学推导

```
步骤 1: 中心化
X_centered = X - mean(X)

步骤 2: 协方差矩阵
Σ = (1/n) × X_centeredᵀ × X_centered

Σᵢⱼ = Cov(特征 i, 特征 j)

对角线：各特征的方差
非对角线：特征间的协方差

步骤 3: 特征分解
Σ × v = λ × v

其中：
- v: 特征向量 (主成分方向)
- λ: 特征值 (方差大小)

步骤 4: 投影
Y = X_centered × W

其中：
- W: 前 K 个特征向量组成的矩阵
- Y: 降维后的数据
```

---

## 四、完整代码实现

### 从零实现 PCA

```python
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components  # 主成分数量
        self.components = None  # 主成分 (特征向量)
        self.mean = None  # 数据均值
        self.explained_variance = None  # 解释方差
        self.explained_variance_ratio = None  # 解释方差比例
    
    def fit(self, X):
        """训练 PCA"""
        n_samples, n_features = X.shape
        
        # 1. 数据中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 3. 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. 排序 (从大到小)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # 5. 选择前 n_components 个主成分
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = (
            eigenvalues[:self.n_components] / np.sum(eigenvalues)
        )
        
        return self
    
    def transform(self, X):
        """投影到主成分空间"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """训练并转换"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """逆变换 (重构原始数据)"""
        return np.dot(X_transformed, self.components.T) + self.mean


# 使用示例
if __name__ == "__main__":
    # 生成 3D 数据
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 0] = X[:, 0] * 3 + X[:, 1] * 2  # 制造相关性
    
    # 训练 PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)
    
    print(f"解释方差比例：{pca.explained_variance_ratio}")
    print(f"累计解释方差：{np.sum(pca.explained_variance_ratio):.4f}")
    
    # 可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 原始 3D 数据
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5)
    ax1.set_title('原始 3D 数据')
    
    # 降维后 2D 数据
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
    ax2.set_title(f'PCA 降维后 (2D)\n解释方差：{np.sum(pca.explained_variance_ratio):.2%}')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    
    # 重构误差
    X_reconstructed = pca.inverse_transform(X_transformed)
    error = np.mean((X - X_reconstructed) ** 2)
    ax3 = fig.add_subplot(133)
    ax3.scatter(X[:, 0], X[:, 1], alpha=0.5, label='原始')
    ax3.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], 
               alpha=0.5, label='重构', marker='x')
    ax3.set_title(f'重构 (MSE={error:.4f})')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('../images/PCA-示例.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 使用 scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 数据标准化 (重要!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建和训练 PCA
pca = PCA(
    n_components=2,      # 降维到 2 维
    svd_solver='auto'    # SVD 求解器
)
X_transformed = pca.fit_transform(X_scaled)

# 查看解释方差
print(f"各主成分解释方差：{pca.explained_variance_ratio_}")
print(f"累计解释方差：{np.sum(pca.explained_variance_ratio_):.4f}")

# 主成分载荷 (特征向量)
print(f"\n主成分载荷:\n{pca.components_}")
```

---

## 五、选择主成分数量

### 解释方差比例

```
每个主成分解释的方差比例：

PC1: 60%  ← 第一主成分解释 60% 的变异
PC2: 25%  ← 第二主成分解释 25% 的变异
PC3: 10%
PC4: 5%

累计：
PC1 + PC2 = 85%  ← 前两个主成分保留 85% 信息
```

### 碎石图 (Scree Plot)

```
思想：画特征值，找"肘部"

特征值
 ↑
 │●
 │ ╲
 │  ╲
 │   ╲___  ← 肘部 (最佳主成分数)
 │       ╲●
 │        ╲
 └────────────→ 主成分
 1  2  3  4  5

肘部之前：重要主成分
肘部之后：噪声
```

### 代码实现

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 计算所有主成分
pca_full = PCA()
pca_full.fit(X_scaled)

# 绘制碎石图
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         pca_full.explained_variance_ratio_, 'o-')
plt.xlabel('主成分')
plt.ylabel('解释方差比例')
plt.title('各主成分解释方差')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), 'o-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% 方差')
plt.xlabel('主成分')
plt.ylabel('累计解释方差')
plt.title('累计解释方差')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/PCA-碎石图.png', dpi=150, bbox_inches='tight')
plt.show()

# 选择保留 95% 方差的主成分数
n_components_95 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1
print(f"保留 95% 方差需要 {n_components_95} 个主成分")
```

### 自动选择

```python
from sklearn.decomposition import PCA

# 自动选择主成分数 (保留 95% 方差)
pca = PCA(
    n_components=0.95,  # 保留 95% 方差
    svd_solver='full'
)
X_transformed = pca.fit_transform(X_scaled)

print(f"自动选择的主成分数：{pca.n_components_}")
print(f"实际保留方差：{np.sum(pca.explained_variance_ratio_):.4f}")
```

---

## 六、PCA 可视化

### 2D 散点图

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 可视化
plt.figure(figsize=(10, 8))

# 按类别着色
for class_label in np.unique(y):
    mask = y == class_label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               label=iris.target_names[class_label],
               alpha=0.7, s=80)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('鸢尾花数据 PCA 降维')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
plt.savefig('../images/PCA-鸢尾花可视化.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 双标图 (Biplot)

```python
def pca_biplot(X_pca, pca, labels, feature_names):
    """绘制 PCA 双标图 (样本点 + 特征向量)"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制样本点
    for class_label in np.unique(labels):
        mask = labels == class_label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  label=f'类别{class_label}',
                  alpha=0.6, s=60)
    
    # 绘制特征向量
    for i, feature in enumerate(feature_names):
        # 特征向量箭头
        arrow_length = 2  # 放大倍数
        ax.arrow(0, 0, 
                pca.components_[0, i] * arrow_length,
                pca.components_[1, i] * arrow_length,
                color='red', alpha=0.8, linewidth=2,
                head_width=0.1)
        # 特征名称
        ax.annotate(feature, 
                   xy=(pca.components_[0, i] * arrow_length * 1.1,
                       pca.components_[1, i] * arrow_length * 1.1),
                   color='red', fontsize=10,
                   ha='center', va='center')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('PCA 双标图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    return fig, ax

# 使用
feature_names = ['特征 1', '特征 2', '特征 3', '特征 4']
fig, ax = pca_biplot(X_pca, pca, y, feature_names)
plt.savefig('../images/PCA-双标图.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 七、PCA 应用场景

### 1. 数据可视化

```python
# 高维数据 → 2D/3D 可视化
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title('高维数据可视化')
plt.show()
```

### 2. 数据压缩

```python
# 保留 95% 信息，减少存储
pca = PCA(n_components=0.95)
X_compressed = pca.fit_transform(X)

# 重构
X_reconstructed = pca.inverse_transform(X_compressed)

# 压缩率
compression_rate = 1 - X_compressed.size / X.size
print(f"压缩率：{compression_rate:.2%}")
```

### 3. 去噪

```python
# 用前几个主成分重构，去除噪声
pca = PCA(n_components=10)
X_denoised = pca.fit_transform(X_noisy)
X_clean = pca.inverse_transform(X_denoised)
```

### 4. 特征提取

```python
# 用主成分作为新特征，训练模型
from sklearn.ensemble import RandomForestClassifier

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_pca, y_train)
print(f"准确率：{clf.score(X_test_pca, y_test):.4f}")
```

### 5. 多重共线性处理

```python
# 相关特征 → 独立主成分
# 解决回归中的共线性问题

from sklearn.linear_model import LinearRegression

pca = PCA()
X_pca = pca.fit_transform(X)

# 用主成分做回归
reg = LinearRegression()
reg.fit(X_pca, y)
```

---

## 八、优缺点

### 优点 ✅

```
✓ 无参数
  - 无需调参
  - 自动计算

✓ 线性方法
  - 计算高效
  - 理论基础强

✓ 去相关
  - 主成分互相独立
  - 消除共线性

✓ 保留最大方差
  - 信息损失最小
  - 最优线性降维

✓ 可解释
  - 特征向量可分析
  - 双标图可视化
```

### 缺点 ❌

```
❌ 线性假设
  - 无法处理非线性结构
  - 流形数据效果差

❌ 方差 = 信息假设
  - 有时重要信息在小方差中
  - 异常值影响大

❌ 需要标准化
  - 量纲影响结果
  - 必须预处理

❌ 可解释性有限
  - 主成分是线性组合
  - 物理意义不直观

❌ 全局方法
  - 忽略局部结构
  - 不如 t-SNE 保局部
```

---

## 九、实战案例：手写数字降维

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
digits = load_digits()
X = digits.data
y = digits.target

print(f"数据形状：{X.shape}")
print(f"原始维度：{X.shape[1]}")

# 2. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA 降维
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 4. 解释方差
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, 'o-')
plt.xlabel('主成分')
plt.ylabel('解释方差比例')
plt.title('各主成分解释方差')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% 方差')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% 方差')
plt.xlabel('主成分')
plt.ylabel('累计解释方差')
plt.title('累计解释方差')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/PCA-数字 - 方差.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. 2D 可视化
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', 
                      alpha=0.6, s=30)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
plt.title('手写数字 PCA 降维 (2D)')
plt.colorbar(scatter, label='数字')
plt.grid(True, alpha=0.3)
plt.savefig('../images/PCA-数字 -2D.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. 3D 可视化
from mpl_toolkits.mplot3d import Axes3D

pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], 
                     c=y, cmap='tab10', alpha=0.6, s=30)
ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
ax.set_title('手写数字 PCA 降维 (3D)')
plt.colorbar(scatter, label='数字')
plt.savefig('../images/PCA-数字 -3D.png', dpi=150, bbox_inches='tight')
plt.show()

# 7. 重构图像
pca_recon = PCA(n_components=10)
X_recon = pca_recon.fit_transform(X_scaled)
X_recon_original = pca_recon.inverse_transform(X_recon)

# 显示原始 vs 重构
fig, axes = plt.subplots(2, 5, figsize=(12, 3))

for i, ax in enumerate(axes[0].flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f'原始')
    ax.axis('off')

for i, ax in enumerate(axes[1].flat):
    ax.imshow(X_recon_original[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'重构 (10 主成分)')
    ax.axis('off')

plt.tight_layout()
plt.savefig('../images/PCA-数字 - 重构.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. 不同主成分数的重构质量
n_components_list = [1, 5, 10, 20, 30, 50]
mse_list = []

for n in n_components_list:
    pca_temp = PCA(n_components=n)
    X_temp = pca_temp.fit_transform(X_scaled)
    X_recon_temp = pca_temp.inverse_transform(X_temp)
    mse = np.mean((X_scaled - X_recon_temp) ** 2)
    mse_list.append(mse)

plt.figure(figsize=(10, 6))
plt.plot(n_components_list, mse_list, 'o-')
plt.xlabel('主成分数量')
plt.ylabel('重构误差 (MSE)')
plt.title('主成分数量 vs 重构误差')
plt.grid(True, alpha=0.3)
plt.savefig('../images/PCA-数字 - 重构误差.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 📝 本章小结

```
┌─────────────────────────────────────────────────────────────┐
│                     核心要点                                │
├─────────────────────────────────────────────────────────────┤
│ • PCA: 线性降维，找到方差最大的方向                         │
│ • 主成分：互相正交，按方差排序                              │
│ • 选择 K: 碎石图、累计方差 (如 95%)                         │
│ • 应用：可视化、压缩、去噪、特征提取                        │
│ • 优点：无参、高效、去相关                                  │
│ • 缺点：线性、方差=信息假设、需标准化                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔜 下一章

[→ 4. t-SNE: 非线性降维可视化](./04-t-SNE.md)

---

*📊 图解机器学习系列 | 无监督学习 第 3 章*
