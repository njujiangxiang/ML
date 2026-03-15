# 4. t-SNE: 非线性降维可视化

## 🎯 学习目标
- 理解 t-SNE 的核心思想
- 掌握 t-SNE 与 PCA 的区别
- 学会 t-SNE 的参数调优
- 能够用 t-SNE 可视化高维数据

---

## 一、为什么需要 t-SNE？

### PCA 的局限

```
PCA (线性降维):
- 假设数据在高维空间是线性分布
- 只能保持全局结构
- 局部邻域关系可能丢失

例子：瑞士卷数据

原始 3D:          PCA 2D:          t-SNE 2D:
  ╱╲              ┌───┐            ┌───┐
 ╱  ╲             │   │            │   │
╱    ╲            └───┘            └───┘
(卷曲流形)        (压扁了)          (展开保持局部)

PCA 把卷压扁，t-SNE 能展开！
```

### t-SNE 的优势

```
t-SNE (t-Distributed Stochastic Neighbor Embedding):

✓ 非线性降维
✓ 保持局部结构 (近邻关系)
✓ 适合可视化 (2D/3D)
✓ 发现流形结构

✗ 计算慢 (O(n²))
✗ 不适合高维输出
✗ 结果有随机性
✗ 只适合可视化，不适合特征提取
```

---

## 二、t-SNE 的核心思想

### 直觉理解

```
核心思想：保持近邻关系

高维空间：              低维空间：
  ●─●                    ●─●
 ╱   ╲                  ╱   ╲
●     ●      →        ●     ●
 ╲   ╱                  ╲   ╱
  ●─●                    ●─●

近的点在近，远的点在远
```

### 算法步骤

```
步骤 1: 高维空间计算相似度

对每个点 i，计算与其他点 j 的条件概率：

p(j|i) = exp(-||xᵢ-xⱼ||² / 2σ²) / Σₖexp(-||xᵢ-xₖ||² / 2σ²)

直觉：
- 距离近 → 概率高
- 距离远 → 概率低
- σ: 控制邻域大小 (困惑度)
```

```
步骤 2: 低维空间计算相似度

在低维空间 (2D/3D)，用 t 分布计算相似度：

q(j|i) = (1 + ||yᵢ-yⱼ||²)⁻¹ / Σₖ(1 + ||yᵢ-yₖ||²)⁻¹

为什么用 t 分布？
- 重尾分布
- 允许远点更远
- 缓解"拥挤问题"
```

```
步骤 3: 最小化分布差异

用 KL 散度衡量两个分布的差异：

KL(P||Q) = ΣᵢΣⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)

用梯度下降优化低维嵌入 Y：
minimize KL(P||Q)

迭代过程：
初始随机 → 逐步调整 → 最终稳定
```

---

## 三、使用 scikit-learn 实现 t-SNE

### 基本用法

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# t-SNE 降维
tsne = TSNE(
    n_components=2,      # 降维到 2D
    perplexity=30,       # 困惑度
    learning_rate=200,   # 学习率
    n_iter=1000,         # 迭代次数
    random_state=42
)

X_tsne = tsne.fit_transform(X)

print(f"原始维度：{X.shape}")
print(f"降维后：{X_tsne.shape}")

# 可视化
plt.figure(figsize=(10, 8))
for class_label in range(3):
    mask = y == class_label
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
               label=iris.target_names[class_label],
               alpha=0.7, s=80)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('鸢尾花数据 t-SNE 可视化')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../images/t-SNE-鸢尾花.png', dpi=150, bbox_inches='tight')
plt.show()
```

### PCA vs t-SNE 对比

```python
from sklearn.decomposition import PCA

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 对比可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
for class_label in range(3):
    mask = y == class_label
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=iris.target_names[class_label],
                   alpha=0.7, s=80)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title(f'PCA (解释方差：{pca.explained_variance_ratio_.sum():.2%})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# t-SNE
for class_label in range(3):
    mask = y == class_label
    axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   label=iris.target_names[class_label],
                   alpha=0.7, s=80)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].set_title('t-SNE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/t-SNE-PCA 对比.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 四、参数调优

### 关键参数

```
┌─────────────────────────────────────────────────────────────┐
│                    t-SNE 关键参数                            │
├──────────────────┬──────────────────────────────────────────┤
│ perplexity       │ 困惑度，控制邻域大小                      │
│                  │ 太小：局部结构，噪声敏感                  │
│                  │ 太大：全局结构，丢失细节                  │
│                  │ 推荐：5-50，常用 30                       │
├──────────────────┼──────────────────────────────────────────┤
│ learning_rate    │ 学习率                                    │
│                  │ 太小：收敛慢，可能陷入局部最优            │
│                  │ 太大：不稳定，可能发散                    │
│                  │ 推荐：100-1000，常用 200                  │
├──────────────────┼──────────────────────────────────────────┤
│ n_iter           │ 迭代次数                                  │
│                  │ 太少：未收敛                              │
│                  │ 太多：浪费时间                            │
│                  │ 推荐：1000-5000                           │
├──────────────────┼──────────────────────────────────────────┤
│ n_components     │ 输出维度                                  │
│                  │ 2 (可视化) 或 3 (3D 可视化)                 │
├──────────────────┼──────────────────────────────────────────┤
│ random_state     │ 随机种子 (保证可重复)                     │
└──────────────────┴──────────────────────────────────────────┘
```

### 困惑度调优

```python
# 尝试不同困惑度
perplexities = [5, 15, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, perp in zip(axes.flat, perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, 
                random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    for class_label in range(3):
        mask = y == class_label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                  label=iris.target_names[class_label],
                  alpha=0.7, s=80)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'Perplexity = {perp}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/t-SNE-困惑度对比.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 五、实战：MNIST 手写数字可视化

```python
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 加载 MNIST 数据
print("加载 MNIST 数据...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int32')

print(f"数据形状：{X.shape}")
print(f"类别数：{len(np.unique(y))}")

# 2. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 先用 PCA 降维到 50 维 (加速 t-SNE)
print("PCA 预降维...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA 后维度：{X_pca.shape}")

# 4. t-SNE 降维 (采样部分数据加速)
print("t-SNE 降维...")
# 采样 5000 个样本
np.random.seed(42)
sample_idx = np.random.choice(len(X_pca), 5000, replace=False)
X_sample = X_pca[sample_idx]
y_sample = y[sample_idx]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=500,
    n_iter=2000,
    random_state=42,
    verbose=1
)
X_tsne = tsne.fit_transform(X_sample)

print(f"t-SNE 完成！最终维度：{X_tsne.shape}")

# 5. 可视化
plt.figure(figsize=(14, 10))

# 每个数字一个颜色
for digit in range(10):
    mask = y_sample == digit
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
               label=str(digit),
               alpha=0.5, s=30)

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('MNIST 手写数字 t-SNE 可视化 (5000 样本)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('../images/t-SNE-MNIST.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. 3D 可视化
from mpl_toolkits.mplot3d import Axes3D

tsne_3d = TSNE(
    n_components=3,
    perplexity=30,
    learning_rate=500,
    n_iter=2000,
    random_state=42
)
X_tsne_3d = tsne_3d.fit_transform(X_sample)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for digit in range(10):
    mask = y_sample == digit
    ax.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], 
               X_tsne_3d[mask, 2],
               label=str(digit),
               alpha=0.5, s=30)

ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')
ax.set_title('MNIST t-SNE 3D 可视化')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('../images/t-SNE-MNIST-3D.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 六、注意事项

### 最佳实践

```
✓ 先 PCA 预降维
  - 降到 30-50 维
  - 加速 t-SNE 计算
  - 去除噪声

✓ 标准化数据
  - t-SNE 基于距离
  - 需要特征尺度一致

✓ 多次运行
  - t-SNE 有随机性
  - 多次运行看稳定性

✓ 困惑度选择
  - 小数据集：5-30
  - 大数据集：30-100
```

### 常见错误

```
✗ 直接用 t-SNE 做特征提取
  → t-SNE 只适合可视化！

✗ 在测试集上拟合 t-SNE
  → t-SNE 不能 transform 新数据！

✗ 困惑度设置太大
  → 丢失局部结构

✗ 迭代次数太少
  → 未收敛，结果不稳定
```

---

## 📝 本章小结

```
┌─────────────────────────────────────────────────────────────┐
│                     核心要点                                │
├─────────────────────────────────────────────────────────────┤
│ • t-SNE: 非线性降维，保持局部结构                           │
│ • 原理：高维相似度 → 低维相似度，最小化 KL 散度              │
│ • 关键参数：困惑度 (5-50)、学习率 (100-1000)                │
│ • 应用：高维数据可视化 (2D/3D)                              │
│ • 最佳实践：先 PCA 预降维，标准化数据                        │
│ • 局限：计算慢、只适合可视化、不能转换新数据                │
└─────────────────────────────────────────────────────────────┘
```

---

## 教程完成！

恭喜！你已经完成了机器学习核心算法的学习！

**回顾已学内容：**
- ✅ 监督学习 (7 章)
- ✅ 无监督学习 (4 章)
- ✅ 模型评估 (3 章)
- ✅ 深度学习 (4 章)

**下一步：**
- 实战项目练习
- Kaggle 竞赛
- 深入特定领域

---

*📊 图解机器学习系列 | 无监督学习 第 4 章 (完结)*
