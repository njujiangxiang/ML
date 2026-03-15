# 1. K-Means 聚类：物以类聚

## 🎯 学习目标
- 理解聚类的核心思想
- 掌握 K-Means 算法原理
- 学会选择最佳 K 值
- 理解 K-Means 的局限性和改进方法

---

## 一、什么是聚类？

### 聚类 vs 分类

```
分类 (监督学习):
  │  ○ ○    ● ●
  │○ ○    ● ●
  │──────────  已知类别，学习边界
  │
  └────────────→
  输入：带标签的数据
  输出：分类模型

聚类 (无监督学习):
  │  ? ?    ? ?
  │? ?    ? ?
  │           没有标签，发现结构
  │
  └────────────→
  输入：无标签的数据
  输出：数据分组
```

### 聚类的目标

```
目标：把相似的东西放在一起

原则：
✓ 组内相似度高 (紧凑)
✓ 组间相似度低 (分离)

可视化：

聚类前：              聚类后：
  │  ● ○ ● ○           │  ● ● ● ○
  │○ ● ○ ●             │○ ○ ○ ●
  │  ● ○ ● ○           │  ● ● ● ○
  └──────────           └──────────
  (混乱)                (分组清晰)
```

---

## 二、K-Means 算法原理

### 核心思想

```
K-Means = 迭代优化，找到 K 个聚类中心

算法步骤：
1. 随机选择 K 个初始中心
2. 将每个点分配到最近的中心
3. 重新计算每个聚类的中心
4. 重复 2-3，直到收敛

可视化：

初始：                  迭代 1:
  │  ●   ●               │  ★   ●
  │●   ●   ●             │●   ★   ●
  │  ★       ★           │  ●       ●
  └──────────           └──────────

迭代 2:                 收敛：
  │  ★   ●               │  ★   ★
  │●   ★   ●             │●   ●   ●
  │  ●       ●           │  ●       ●
  └──────────           └──────────
  (中心移动)            (稳定)
```

### 详细步骤

```
步骤 1: 初始化
  │
  │  ●   ●   ●
  │●   ★   ●   ★  ← 随机选 K 个点作为中心
  │  ●   ●   ●
  └──────────

步骤 2: 分配
  │
  │  ●   ●   ●
  │●  ╱ ╲  ●   ●  ← 每个点分配到最近的中心
  │  ●   ●   ●
  └──────────

步骤 3: 更新中心
  │
  │  ●   ●   ●
  │●   ☆   ●   ★  ← 中心移动到聚类平均位置
  │  ●   ●   ●
  └──────────

步骤 4: 重复
  回到步骤 2，直到中心不再移动
```

---

## 三、数学原理

### 目标函数

```
K-Means 优化目标：

J = Σ Σ ||xᵢ - μₖ||²
    k=1 i∈Cₖ

其中：
- K: 聚类数量
- Cₖ: 第 k 个聚类的点
- μₖ: 第 k 个聚类的中心
- ||xᵢ - μₖ||²: 点到中心的距离平方

目标：最小化 J
即：让每个点尽可能靠近所属聚类中心
```

### 算法收敛

```
K-Means 保证收敛：
✓ 每次迭代 J 都减小
✓ J 有下界 (≥0)
✓ 所以必然收敛

但：
❌ 可能收敛到局部最优
❌ 不一定是最优解

解决：
✓ 多次随机初始化
✓ 选 J 最小的结果
```

---

## 四、完整代码实现

### 从零实现 K-Means

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
    
    def _initialize_centroids(self, X):
        """随机初始化聚类中心"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.k, replace=False)
        return X[indices]
    
    def _assign_clusters(self, X, centroids):
        """将每个点分配到最近的聚类中心"""
        distances = np.zeros((X.shape[0], self.k))
        
        for i in range(self.k):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """更新聚类中心"""
        centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            points = X[labels == i]
            if len(points) > 0:
                centroids[i] = points.mean(axis=0)
            else:
                # 如果聚类为空，随机选择一个点
                centroids[i] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """计算惯性 (目标函数值)"""
        inertia = 0
        for i in range(self.k):
            points = X[labels == i]
            if len(points) > 0:
                inertia += np.sum((points - centroids[i]) ** 2)
        return inertia
    
    def fit(self, X):
        """训练 K-Means"""
        # 初始化
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # 分配
            self.labels = self._assign_clusters(X, self.centroids)
            
            # 更新
            new_centroids = self._update_centroids(X, self.labels)
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                print(f"迭代 {iteration + 1} 次后收敛")
                break
            
            self.centroids = new_centroids
        
        # 计算最终惯性
        self.inertia = self._compute_inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """预测新样本的聚类"""
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """训练并预测"""
        self.fit(X)
        return self.labels


# 使用示例
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    # 生成数据
    X, y = make_blobs(n_samples=300, centers=4, 
                      cluster_std=0.60, random_state=42)
    
    # 训练 K-Means
    model = KMeans(k=4, random_state=42)
    labels = model.fit_predict(X)
    
    print(f"惯性：{model.inertia:.2f}")
    
    # 可视化
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    plt.title('真实标签')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], 
               c='red', s=200, marker='X', label='聚类中心')
    plt.title(f'K-Means 聚类 (K=4)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../images/KMeans-示例.png', dpi=150, bbox_inches='tight')
    plt.show()
```

### 使用 scikit-learn

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成数据
X, y = make_blobs(n_samples=300, centers=4, 
                  cluster_std=0.60, random_state=42)

# 创建和训练模型
kmeans = KMeans(
    n_clusters=4,       # 聚类数量 K
    init='k-means++',   # 初始化方法
    n_init=10,          # 运行次数
    max_iter=300,       # 最大迭代次数
    random_state=42
)
kmeans.fit(X)

# 预测
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"聚类中心:\n{centroids}")
print(f"惯性：{inertia:.2f}")
```

---

## 五、选择最佳 K 值

### 肘部法则 (Elbow Method)

```
思想：K 增大，惯性减小；找"肘部"点

惯性
 ↑
 │╲
 │ ╲
 │  ╲
 │   ╲___  ← 肘部 (最佳 K)
 │       ╲
 │        ╲
 └────────────→ K
 1  2  3  4  5  6

肘部之前：K 增加，惯性大幅下降
肘部之后：K 增加，惯性下降缓慢
```

### 代码实现

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 计算不同 K 值的惯性
k_range = range(1, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 绘制肘部图
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'o-')
plt.xlabel('聚类数量 K')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.grid(True, alpha=0.3)
plt.savefig('../images/KMeans-肘部法则.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 轮廓系数 (Silhouette Score)

```
轮廓系数 = 衡量聚类质量

s(i) = (b(i) - a(i)) / max(a(i), b(i))

其中：
- a(i): 点 i 到同聚类其他点的平均距离
- b(i): 点 i 到最近其他聚类点的平均距离

s(i) 范围：[-1, 1]
- s ≈ 1: 聚类好
- s ≈ 0: 聚类重叠
- s ≈ -1: 聚类错误

平均轮廓系数越高，聚类越好
```

### 代码实现

```python
from sklearn.metrics import silhouette_score

# 计算不同 K 值的轮廓系数
k_range = range(2, 11)  # K=1 时轮廓系数无定义
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}, 轮廓系数={score:.4f}")

# 绘制
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('聚类数量 K')
plt.ylabel('轮廓系数')
plt.title('轮廓系数法')
plt.grid(True, alpha=0.3)
plt.savefig('../images/KMeans-轮廓系数.png', dpi=150, bbox_inches='tight')
plt.show()

# 最佳 K
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n最佳 K 值：{best_k}")
```

---

## 六、K-Means++ 初始化

### 问题：随机初始化可能不好

```
坏初始化：
  │  ★ ★       ● ● ●  ← 两个中心太近
  │● ● ●   ● ● ●
  │
  └────────────
  收敛慢，可能陷入局部最优

好初始化：
  │  ★       ● ● ●
  │● ● ●   ● ● ●
  │              ★
  └────────────
  中心分散，收敛快
```

### K-Means++ 解决方案

```
K-Means++ 初始化：
1. 随机选第一个中心
2. 对每个点，计算到最近中心的距离 D(x)
3. 按 D(x)² 的概率选择下一个中心
   (距离越远，概率越大)
4. 重复 2-3，直到选满 K 个

好处：
✓ 中心分散
✓ 收敛更快
✓ 结果更好
```

### 代码实现

```python
from sklearn.cluster import KMeans

# 使用 K-Means++ 初始化 (默认)
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',  # K-Means++ 初始化
    n_init=10          # 运行 10 次选最好的
)
kmeans.fit(X)

# 随机初始化 (不推荐)
kmeans_random = KMeans(
    n_clusters=4,
    init='random',     # 随机初始化
    n_init=10
)
kmeans_random.fit(X)

print(f"K-Means++ 惯性：{kmeans.inertia_:.2f}")
print(f"随机初始化惯性：{kmeans_random.inertia_:.2f}")
```

---

## 七、局限性和改进

### 局限性

```
❌ 需要指定 K
  - 事先不知道聚类数
  - 需要尝试多个 K

❌ 假设球形聚类
  │  ● ●              │  ●●
  │●    ●    ●●       │●  ●  ●●
  │ ● ●      ●●       │●●
  └──────────          └──────────
  (球形，好)           (非球形，差)

❌ 对异常值敏感
  │  ● ● ●
  │●   ●   ●
  │  ● ● ●      ★  ← 异常值
  └────────────
  中心被拉偏

❌ 大小不均匀的聚类效果差
  │●●●●●●●●  ○○
  │●●●●●●●●  ○○
  └────────────
  小聚类可能被忽略
```

### 改进方法

```
1. K-Means++
   - 改进初始化
   - 已集成在 scikit-learn

2. Mini-Batch K-Means
   - 用数据子集更新
   - 适合大数据集
   - 更快

3. K-Medoids
   - 用实际点作为中心
   - 对异常值鲁棒

4. 层次聚类
   - 不需要指定 K
   - 可以处理任意形状

5. DBSCAN
   - 基于密度
   - 可以发现任意形状
   - 自动识别噪声
```

### Mini-Batch K-Means

```python
from sklearn.cluster import MiniBatchKMeans

# Mini-Batch K-Means
mb_kmeans = MiniBatchKMeans(
    n_clusters=4,
    batch_size=100,    # 每批样本数
    n_init=10,
    random_state=42
)
mb_kmeans.fit(X)

print(f"Mini-Batch 惯性：{mb_kmeans.inertia_:.2f}")
print(f"训练时间更短！")
```

---

## 八、实战案例：客户细分

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
data = pd.read_csv('customers.csv')
print(f"数据量：{len(data)}")
print(f"\n特征:\n{data.columns.tolist()}")

# 2. 数据探索
print(f"\n数据描述:\n{data.describe()}")

# 3. 数据预处理
# 删除 ID 列
if 'CustomerID' in data.columns:
    data = data.drop('CustomerID', axis=1)

# 处理缺失值
data = data.dropna()

# 4. 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 5. 选择最佳 K
from sklearn.metrics import silhouette_score

k_range = range(2, 11)
inertias = []
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(k_range, inertias, 'o-')
axes[0].set_xlabel('K 值')
axes[0].set_ylabel('惯性')
axes[0].set_title('肘部法则')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_range, silhouettes, 'o-')
axes[1].set_xlabel('K 值')
axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/KMeans-客户细分 -K 选择.png', dpi=150, bbox_inches='tight')
plt.show()

# 6. 训练最终模型
best_k = 4  # 根据上面的图选择
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# 7. 分析聚类特征
data['Cluster'] = labels

# 各聚类的特征均值
cluster_profile = data.groupby('Cluster').mean()
print(f"\n聚类特征:\n{cluster_profile}")

# 可视化
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_profile.T, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('各聚类特征画像')
plt.xlabel('聚类')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('../images/KMeans-客户画像.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. PCA 降维可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 聚类中心的 PCA 投影
centroids_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                      cmap='viridis', alpha=0.5, s=50)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='red', s=300, marker='X', label='聚类中心')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('客户聚类 (PCA 降维)')
plt.legend()
plt.colorbar(scatter, label='聚类')
plt.savefig('../images/KMeans-客户聚类可视化.png', dpi=150, bbox_inches='tight')
plt.show()

# 9. 聚类解释
print("\n=== 聚类解释 ===")
for i in range(best_k):
    print(f"\n聚类 {i}:")
    print(cluster_profile.loc[i].sort_values(ascending=False).head(3))
```

---

## 📝 本章小结

```
┌─────────────────────────────────────────────────────────────┐
│                     核心要点                                │
├─────────────────────────────────────────────────────────────┤
│ • K-Means: 迭代优化，找到 K 个聚类中心                       │
│ • 目标：最小化点到聚类中心的距离平方和                      │
│ • K 值选择：肘部法则、轮廓系数                              │
│ • K-Means++: 更好的初始化方法                               │
│ • 局限：需要指定 K、假设球形、对异常值敏感                  │
│ • 改进：Mini-Batch K-Means、K-Medoids、DBSCAN              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔜 下一章

[→ 2. 层次聚类：树状分组](./02-层次聚类.md)

---

*📊 图解机器学习系列 | 无监督学习 第 1 章*
