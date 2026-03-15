"""
机器学习教程 - 可视化示例生成脚本
生成各种算法的示意图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片目录
import os
os.makedirs('../images', exist_ok=True)

print("开始生成可视化图片...")

# ============================================
# 1. 线性回归示意图
# ============================================
print("1. 生成线性回归示意图...")

np.random.seed(42)
X = np.linspace(0, 10, 50)
y_true = 2 * X + 1
y_noise = y_true + np.random.randn(50) * 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 欠拟合
axes[0].scatter(X, y_noise, alpha=0.5, label='数据点')
axes[0].plot(X, np.ones(50) * np.mean(y_noise), 'r-', linewidth=2, label='欠拟合')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('欠拟合 (太简单)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 刚好
axes[1].scatter(X, y_noise, alpha=0.5, label='数据点')
axes[1].plot(X, y_true, 'r-', linewidth=2, label='最佳拟合')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('刚好 (合适)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 过拟合
axes[2].scatter(X, y_noise, alpha=0.5, label='数据点')
X_smooth = np.linspace(0, 10, 200)
y_poly = np.poly1d(np.polyfit(X, y_noise, 10))(X_smooth)
axes[2].plot(X_smooth, y_poly, 'r-', linewidth=2, label='过拟合')
axes[2].set_xlabel('X')
axes[2].set_ylabel('y')
axes[2].set_title('过拟合 (太复杂)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/01-线性回归 - 拟合程度对比.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 2. 梯度下降可视化
# ============================================
print("2. 生成梯度下降示意图...")

def loss_function(w):
    return (w - 3) ** 2 + 2

w = np.linspace(-2, 8, 100)
loss = loss_function(w)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失函数曲面
axes[0].plot(w, loss, 'b-', linewidth=2)
axes[0].fill_between(w, loss, alpha=0.3)
axes[0].plot(3, 2, 'r*', markersize=15, label='最小值点')

# 梯度下降路径
w_path = [-1, 0.5, 1.5, 2.2, 2.6, 2.85, 3]
for i in range(len(w_path) - 1):
    axes[0].annotate('', xy=(w_path[i+1], loss_function(w_path[i+1])), 
                    xytext=(w_path[i], loss_function(w_path[i])),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    axes[0].plot(w_path[i], loss_function(w_path[i]), 'ro')

axes[0].set_xlabel('参数 w')
axes[0].set_ylabel('损失')
axes[0].set_title('梯度下降过程')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 学习率对比
axes[1].plot(w, loss, 'b-', linewidth=2, label='损失函数')

# 学习率太小
w_small = np.linspace(-1, 2.5, 20)
loss_small = loss_function(w_small)
axes[1].plot(w_small, loss_small, 'g-o', label='学习率太小', alpha=0.5)

# 学习率刚好
w_good = [-1, 1, 2, 2.5, 2.75, 2.9]
loss_good = loss_function(np.array(w_good))
axes[1].plot(w_good, loss_good, 'r-o', label='学习率刚好')

# 学习率太大
w_large = [-1, 5, 0, 4, 1]
loss_large = loss_function(np.array(w_large))
axes[1].plot(w_large, loss_large, 'm-o', label='学习率太大', alpha=0.5)

axes[1].plot(3, 2, 'k*', markersize=15)
axes[1].set_xlabel('参数 w')
axes[1].set_ylabel('损失')
axes[1].set_title('学习率的影响')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/02-梯度下降示意图.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 3. Sigmoid 函数
# ============================================
print("3. 生成 Sigmoid 函数示意图...")

z = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z, sigmoid, 'b-', linewidth=3)
ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax.plot(0, 0.5, 'ro', markersize=10)
ax.annotate('(0, 0.5)', xy=(0, 0.5), xytext=(1, 0.6),
           arrowprops=dict(arrowstyle='->', color='red'))

ax.set_xlabel('z', fontsize=12)
ax.set_ylabel('σ(z)', fontsize=12)
ax.set_title('Sigmoid 函数', fontsize=14)
ax.set_ylim(-0.1, 1.1)
ax.grid(True, alpha=0.3)
ax.set_aspect('auto')

plt.tight_layout()
plt.savefig('../images/03-sigmoid 函数.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 4. 逻辑回归决策边界
# ============================================
print("4. 生成逻辑回归决策边界示意图...")

np.random.seed(42)
n_samples = 100

# 生成两类数据
X_class0 = np.random.randn(n_samples//2, 2) * 0.8 + np.array([-1.5, -1.5])
X_class1 = np.random.randn(n_samples//2, 2) * 0.8 + np.array([1.5, 1.5])

X = np.vstack([X_class0, X_class1])
y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])

fig, ax = plt.subplots(figsize=(8, 6))

# 绘制决策边界
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))

# 简单线性边界
decision_boundary = -(xx1 + xx2) / 2
Z = (xx1 + xx2) / 2

ax.contourf(xx1, xx2, Z, levels=[-np.inf, 0, np.inf], 
            colors=['lightblue', 'lightcoral'], alpha=0.3)
ax.contour(xx1, xx2, Z, levels=[0], colors='red', linewidths=2)

ax.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', 
          label='类别 0', alpha=0.6, edgecolors='k', s=50)
ax.scatter(X_class1[:, 0], X_class1[:, 1], c='red', 
          label='类别 1', alpha=0.6, edgecolors='k', s=50)

ax.set_xlabel('特征 1')
ax.set_ylabel('特征 2')
ax.set_title('逻辑回归决策边界')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('../images/04-逻辑回归决策边界.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 5. 交叉熵损失函数
# ============================================
print("5. 生成交叉熵损失函数示意图...")

p = np.linspace(0.01, 0.99, 100)

# y = 1 时的损失
loss_y1 = -np.log(p)

# y = 0 时的损失
loss_y0 = -np.log(1 - p)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# y = 1
axes[0].plot(p, loss_y1, 'b-', linewidth=2)
axes[0].fill_between(p, loss_y1, alpha=0.3)
axes[0].set_xlabel('预测概率 p')
axes[0].set_ylabel('损失')
axes[0].set_title('真实标签 y = 1\n预测越接近 1，损失越小')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 5)

# y = 0
axes[1].plot(p, loss_y0, 'r-', linewidth=2)
axes[1].fill_between(p, loss_y0, alpha=0.3)
axes[1].set_xlabel('预测概率 p')
axes[1].set_ylabel('损失')
axes[1].set_title('真实标签 y = 0\n预测越接近 0，损失越小')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 5)

plt.tight_layout()
plt.savefig('../images/05-交叉熵损失函数.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 6. 混淆矩阵示例
# ============================================
print("6. 生成混淆矩阵示意图...")

confusion_matrix = np.array([[85, 15],
                             [10, 90]])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测负例', '预测正例'],
            yticklabels=['实际负例', '实际正例'],
            ax=ax, annot_kws={'size': 14})

ax.set_xlabel('预测', fontsize=12)
ax.set_ylabel('实际', fontsize=12)
ax.set_title('混淆矩阵', fontsize=14)

# 添加 TP, TN, FP, FN 标注
ax.text(0.75, 0.25, 'TN', ha='center', va='center', 
        fontsize=10, color='gray', alpha=0.5)
ax.text(1.75, 0.25, 'FP', ha='center', va='center', 
        fontsize=10, color='gray', alpha=0.5)
ax.text(0.75, 1.25, 'FN', ha='center', va='center', 
        fontsize=10, color='gray', alpha=0.5)
ax.text(1.75, 1.25, 'TP', ha='center', va='center', 
        fontsize=10, color='gray', alpha=0.5)

plt.tight_layout()
plt.savefig('../images/06-混淆矩阵示例.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 7. ROC 曲线
# ============================================
print("7. 生成 ROC 曲线示意图...")

# 模拟 ROC 曲线数据
fpr_perfect = [0, 0, 1]
tpr_perfect = [0, 1, 1]

fpr_good = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1]
tpr_good = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1]

fpr_random = [0, 1]
tpr_random = [0, 1]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fpr_perfect, tpr_perfect, 'g-', linewidth=2, label='完美分类器 (AUC=1.0)')
ax.plot(fpr_good, tpr_good, 'b-', linewidth=2, label='好模型 (AUC≈0.85)')
ax.plot(fpr_random, tpr_random, 'k--', linewidth=2, label='随机猜测 (AUC=0.5)')

ax.fill_between(fpr_good, tpr_good, alpha=0.3, color='blue')
ax.set_xlabel('假正例率 (FPR)', fontsize=12)
ax.set_ylabel('真正例率 (TPR)', fontsize=12)
ax.set_title('ROC 曲线', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('../images/07-ROC 曲线示例.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 8. 正则化效果对比
# ============================================
print("8. 生成正则化效果对比图...")

np.random.seed(42)
X = np.linspace(0, 10, 20)
y_true = np.sin(X)
y_noise = y_true + np.random.randn(20) * 0.3

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 无正则化 (过拟合)
coef_no_reg = np.polyfit(X, y_noise, 10)
y_pred_no_reg = np.poly1d(coef_no_reg)(np.linspace(0, 10, 200))
axes[0].scatter(X, y_noise, alpha=0.5, label='数据点')
axes[0].plot(np.linspace(0, 10, 200), y_pred_no_reg, 'r-', linewidth=2, label='无正则化')
axes[0].plot(np.linspace(0, 10, 200), np.sin(np.linspace(0, 10, 200)), 'g--', alpha=0.5, label='真实函数')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title('无正则化 (过拟合)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# L2 正则化 (合适)
coef_l2 = np.polyfit(X, y_noise, 5)
y_pred_l2 = np.poly1d(coef_l2)(np.linspace(0, 10, 200))
axes[1].scatter(X, y_noise, alpha=0.5, label='数据点')
axes[1].plot(np.linspace(0, 10, 200), y_pred_l2, 'b-', linewidth=2, label='L2 正则化')
axes[1].plot(np.linspace(0, 10, 200), np.sin(np.linspace(0, 10, 200)), 'g--', alpha=0.5, label='真实函数')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title('L2 正则化 (合适)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 强正则化 (欠拟合)
coef_strong = np.polyfit(X, y_noise, 2)
y_pred_strong = np.poly1d(coef_strong)(np.linspace(0, 10, 200))
axes[2].scatter(X, y_noise, alpha=0.5, label='数据点')
axes[2].plot(np.linspace(0, 10, 200), y_pred_strong, 'm-', linewidth=2, label='强正则化')
axes[2].plot(np.linspace(0, 10, 200), np.sin(np.linspace(0, 10, 200)), 'g--', alpha=0.5, label='真实函数')
axes[2].set_xlabel('X')
axes[2].set_ylabel('y')
axes[2].set_title('强正则化 (欠拟合)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/08-正则化效果对比.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 9. 机器学习工作流程图
# ============================================
print("9. 生成机器学习工作流程图...")

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')

# 步骤
steps = ['收集数据', '数据预处理', '选择模型', '训练模型', '评估模型', '部署应用']
icons = ['📊', '🧹', '🤖', '📚', '📝', '🚀']

for i, (step, icon) in enumerate(zip(steps, icons)):
    # 绘制方框
    rect = plt.Rectangle((i * 2, 0.5), 1.5, 1, fill=True, 
                         facecolor=f'C{i}', alpha=0.3, 
                         edgecolor=f'C{i}', linewidth=2)
    ax.add_patch(rect)
    
    # 添加文字
    ax.text(i * 2 + 0.75, 1.2, f'{icon}\n{step}', 
            ha='center', va='center', fontsize=11,
            fontweight='bold')
    
    # 添加箭头
    if i < len(steps) - 1:
        ax.annotate('', xy=(i * 2 + 1.5, 1), 
                   xytext=(i * 2 + 1.9, 1),
                   arrowprops=dict(arrowstyle='->', 
                                  color='gray', 
                                  linewidth=2))

ax.set_xlim(-0.5, len(steps) * 2)
ax.set_ylim(0, 2)
ax.set_aspect('equal')
ax.set_title('机器学习完整工作流程', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig('../images/09-机器学习工作流程图.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# 10. 监督学习 vs 无监督学习
# ============================================
print("10. 生成监督学习 vs 无监督学习对比图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 监督学习
np.random.seed(42)
X_sup = np.random.randn(100, 2)
y_sup = (X_sup[:, 0] + X_sup[:, 1] > 0).astype(int)

axes[0].scatter(X_sup[y_sup==0, 0], X_sup[y_sup==0, 1], 
               c='blue', label='类别 0', alpha=0.6, edgecolors='k')
axes[0].scatter(X_sup[y_sup==1, 0], X_sup[y_sup==1, 1], 
               c='red', label='类别 1', alpha=0.6, edgecolors='k')
axes[0].plot([-2, 2], [0, 0], 'k-', linewidth=2, label='决策边界')
axes[0].set_xlabel('特征 1')
axes[0].set_ylabel('特征 2')
axes[0].set_title('监督学习 (有标签)\n学习分类边界')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# 无监督学习
np.random.seed(42)
X_unsup = np.vstack([
    np.random.randn(50, 2) * 0.5 + np.array([-1, -1]),
    np.random.randn(50, 2) * 0.5 + np.array([1, 1]),
    np.random.randn(50, 2) * 0.5 + np.array([-1, 1])
])

# K-Means 聚类结果
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_unsup)

colors = ['red', 'blue', 'green']
for i in range(3):
    axes[1].scatter(X_unsup[labels==i, 0], X_unsup[labels==i, 1], 
                   c=colors[i], label=f'簇{i+1}', alpha=0.6, edgecolors='k')

# 聚类中心
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='yellow', s=200, marker='X', edgecolors='k', linewidths=2,
               label='聚类中心')

axes[1].set_xlabel('特征 1')
axes[1].set_ylabel('特征 2')
axes[1].set_title('无监督学习 (无标签)\n发现数据分组')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('../images/10-监督学习 vs 无监督学习.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ 所有图片生成完成！")
print("图片保存在：../images/ 目录下")
