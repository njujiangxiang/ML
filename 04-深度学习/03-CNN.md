# 3. 卷积神经网络 (CNN): 图像识别利器

## 🎯 学习目标
- 理解 CNN 为什么适合图像处理
- 掌握卷积层、池化层的工作原理
- 理解经典 CNN 架构
- 能够用代码实现简单 CNN

---

## 一、为什么用 CNN 处理图像？

### 全连接网络的问题

```
全连接网络处理图像：

28×28 图像 → 展平 → 784 维向量
         ↓
    全连接层 (权重：784×128 = 100,352)

问题：
✗ 参数量大 (高分辨率更严重)
  1000×1000 图像 → 100 万输入 → 1 亿权重！

✗ 忽略空间结构
  相邻像素的关系被破坏

✗ 平移不不变性
  物体移动位置，识别失败
  
  ●●●              ●●●
  ●  ●    →   ●  ●    (同一个物体，位置不同)
  ●●●              ●●●
```

### CNN 的优势

```
CNN 核心思想：

1. 局部连接
   神经元只连接局部区域 (感受野)
   
   ╔═══╗
   ║███║  ← 只连接 3×3 区域
   ╚═══╝

2. 权值共享
   同一个卷积核扫过整张图
   
   ╔═══════╗
   ║█→→→→║
   ║↓     ║  ← 同一组权重
   ║↓     ║
   ╚═══════╝

3. 空间层次结构
   浅层：边缘、纹理
   中层：形状、部件
   深层：物体、场景
   
   边缘 → 眼睛 → 人脸
```

---

## 二、卷积层：特征提取器

### 卷积操作

```
输入图像 (5×5) + 卷积核 (3×3) → 输出特征图 (3×3)

输入：          卷积核：        输出：
┌─────┐        ┌───┐          ┌───┐
│1 1 1│        │1 0│          │ 2 │
│0 1 0│   *    │0 1│   =      │   │
│1 0 1│        │1 0│          │   │
│0 1 0│        └───┘          └───┘
│1 1 1│
└─────┘

计算过程 (第一步):
┌───┐
│1 1 1│
│0 1 0│  ← 左上角 3×3 区域
│1 0 1│
└───┘

1×1 + 1×0 + 1×1 +
0×0 + 1×1 + 0×0 +
1×1 + 0×0 + 1×0 = 1+0+1+0+1+0+1+0+0 = 3
```

### 卷积参数

```
1. 卷积核大小 (Kernel Size)
   常见：3×3, 5×5, 7×7
   
   小卷积核：
   ✓ 参数少
   ✓ 可以堆叠多层
   ✓ 感受野逐步增大

2. 步长 (Stride)
   每次移动的步数
   
   stride=1: 每次移 1 格 (输出大)
   stride=2: 每次移 2 格 (输出减半)

3. 填充 (Padding)
   是否在边缘补 0
   
   Valid: 不填充 (输出变小)
   Same:  填充保持大小

4. 卷积核数量
   输出特征图的通道数
   
   64 个卷积核 → 64 个特征图
```

### 输出尺寸计算

```
输出尺寸 = (输入尺寸 - 卷积核大小 + 2×填充) / 步长 + 1

例子：
输入：32×32
卷积核：3×3
填充：1
步长：1

输出 = (32 - 3 + 2×1) / 1 + 1 = 32×32 (保持不变)
```

### 代码实现

```python
import numpy as np

def conv2d(input_tensor, kernel, stride=1, padding=0):
    """
    2D 卷积 (简化版，单通道)
    
    input_tensor: 输入 (H, W)
    kernel: 卷积核 (kH, kW)
    stride: 步长
    padding: 填充
    """
    # 填充
    if padding > 0:
        input_tensor = np.pad(input_tensor, padding, mode='constant')
    
    h_in, w_in = input_tensor.shape
    kh, kw = kernel.shape
    
    # 计算输出尺寸
    h_out = (h_in - kh) // stride + 1
    w_out = (w_in - kw) // stride + 1
    
    # 输出
    output = np.zeros((h_out, w_out))
    
    # 卷积操作
    for i in range(h_out):
        for j in range(w_out):
            region = input_tensor[i*stride:i*stride+kh, 
                                  j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

# 测试
if __name__ == "__main__":
    # 输入图像
    input_img = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0]
    ])
    
    # 卷积核 (边缘检测)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
    
    # 卷积
    output = conv2d(input_img, kernel, stride=1, padding=0)
    
    print("输入:")
    print(input_img)
    print("\n卷积核 (垂直边缘检测):")
    print(kernel)
    print("\n输出:")
    print(output)
```

### 多通道卷积

```
RGB 图像 (H×W×3) + 卷积核 (3×3×3) → 输出 (H'×W'×1)

输入：
┌─────┐
│ R   │  红色通道
│ G   │  绿色通道
│ B   │  蓝色通道
└─────┘

卷积核：
┌───┐
│K_R│  红色通道核
│K_G│  绿色通道核
│K_B│  蓝色通道核
└───┘

输出：
每个位置的 3 个通道分别卷积后相加
```

---

## 三、池化层：降维和下采样

### 最大池化 (Max Pooling)

```
取区域内的最大值

输入 (4×4)          输出 (2×2)
┌───────┐          ┌───┐
│1 3│2 1│          │ 4 │
│2 4│1 2│          │   │
├───┼───┤    →     │   │
│1 2│3 5│          │ 5 │
│0 1│2 4│          │   │
└───────┘          └───┘

2×2 池化，步长 2
每个 2×2 区域取最大值
```

### 平均池化 (Average Pooling)

```
取区域内的平均值

输入 (4×4)          输出 (2×2)
┌───────┐          ┌───┐
│1 3│2 1│          │2.5│
│2 4│1 2│          │   │
├───┼───┤    →     │   │
│1 2│3 5│          │3.0│
│0 1│2 4│          │   │
└───────┘          └───┘

(1+3+2+4)/4 = 2.5
(1+2+3+5)/4 = 2.75 ≈ 3.0
```

### 池化的作用

```
✓ 降低维度
  减少计算量

✓ 扩大感受野
  后续层看到更大区域

✓ 平移不变性
  小移动不影响输出
  
  ●●    ●●
  ●● → ●●  (最大池化后相同)

✓ 防止过拟合
  减少参数
```

### 代码实现

```python
def max_pooling(input_tensor, pool_size=2, stride=2):
    """最大池化"""
    h_in, w_in = input_tensor.shape
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input_tensor[i*stride:i*stride+pool_size,
                                  j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    
    return output

# 测试
input_img = np.array([
    [1, 3, 2, 1],
    [2, 4, 1, 2],
    [1, 2, 3, 5],
    [0, 1, 2, 4]
])

output = max_pooling(input_img, pool_size=2, stride=2)
print("输入:")
print(input_img)
print("\n最大池化输出:")
print(output)
```

---

## 四、经典 CNN 架构

### LeNet-5 (1998)

```
最早的 CNN 之一，用于手写数字识别

输入 (32×32×1)
  ↓
卷积 (6@28×28)  ← 6 个 5×5 卷积核
  ↓
池化 (6@14×14)  ← 2×2 池化
  ↓
卷积 (16@10×10)
  ↓
池化 (16@5×5)
  ↓
全连接 (120)
  ↓
全连接 (84)
  ↓
输出 (10)  ← 0-9 数字

参数量：~60K
```

### AlexNet (2012)

```
ImageNet 竞赛冠军，深度学习爆发点

输入 (227×227×3)
  ↓
卷积 (96@55×55), 11×11, stride=4
  ↓
池化 (96@27×27)
  ↓
卷积 (256@27×27), 5×5
  ↓
池化 (256@13×13)
  ↓
卷积 (384@13×13), 3×3
  ↓
卷积 (384@13×13), 3×3
  ↓
卷积 (256@13×13), 3×3
  ↓
池化 (256@6×6)
  ↓
全连接 (4096)
  ↓
全连接 (4096)
  ↓
输出 (1000)  ← ImageNet 1000 类

创新：
✓ ReLU 激活
✓ Dropout
✓ GPU 训练
✓ 数据增强
```

### VGG (2014)

```
用多个小卷积核替代大卷积核

VGG-16:
输入 (224×224×3)
  ↓
[卷积 (64@224×224), 3×3] × 2
  ↓
池化
  ↓
[卷积 (128@112×112), 3×3] × 2
  ↓
池化
  ↓
[卷积 (256@56×56), 3×3] × 3
  ↓
池化
  ↓
[卷积 (512@28×28), 3×3] × 3
  ↓
池化
  ↓
[卷积 (512@14×14), 3×3] × 3
  ↓
池化
  ↓
全连接 (4096)
  ↓
全连接 (4096)
  ↓
输出 (1000)

特点：
✓ 统一用 3×3 卷积
✓ 结构规整
✓ 参数量大 (138M)
```

### ResNet (2015)

```
残差网络，解决梯度消失

核心：残差块

输入 x
  ↓      ┌──────────┐
卷积     │          │
  ↓      │          │
ReLU     │          │
  ↓      │          │
卷积     ↓          │
  ↓    相加 (x+F(x))
BatchNorm  ↓
  ↓      │
ReLU     │
  ↓      │
输出      └──────────┘

F(x) + x  ← 残差连接

好处：
✓ 梯度可以直接传回
✓ 可以训练很深 (152 层)
✓ 性能更好
```

---

## 五、用 Keras 实现 CNN

```python
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. 加载数据 (CIFAR-10)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(f"训练集：{x_train.shape}")  # (50000, 32, 32, 3)
print(f"测试集：{x_test.shape}")    # (10000, 32, 32, 3)

# 2. 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot 编码
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# 3. 构建 CNN 模型
model = keras.Sequential([
    # 卷积块 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # 卷积块 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # 卷积块 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # 全连接层
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型结构
model.summary()

# 5. 训练
history = model.fit(
    x_train, y_train_cat,
    batch_size=64,
    epochs=50,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)

# 6. 评估
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\n测试准确率：{test_acc:.4f}")

# 7. 可视化训练过程
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='训练')
axes[0].plot(history.history['val_accuracy'], label='验证')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('准确率')
axes[0].set_title('训练准确率')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='训练')
axes[1].plot(history.history['val_loss'], label='验证')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('损失')
axes[1].set_title('训练损失')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/CNN-训练过程.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. 预测示例
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', 
               '狗', '青蛙', '马', '船', '卡车']

predictions = model.predict(x_test[:10])
predicted_labels = predictions.argmax(axis=1)

fig, axes = plt.subplots(2, 5, figsize=(15, 3))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i])
    ax.set_title(f'预测：{class_names[predicted_labels[i]]}\n'
                f'真实：{class_names[y_test[i][0]]}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('../images/CNN-预测示例.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 六、数据增强

```
增加训练数据，防止过拟合

常见增强方法：
✓ 随机翻转
✓ 随机旋转
✓ 随机裁剪
✓ 颜色抖动
✓ 平移缩放
```

### Keras 数据增强

```python
from tensorflow.keras import layers

# 数据增强层
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# 在模型中使用
model = keras.Sequential([
    data_augmentation,  # 先增强
    layers.Conv2D(32, 3, activation='relu'),
    # ... 其他层
])

# 或者在训练时使用
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    callbacks=[data_augmentation]  # 仅训练时应用
)
```

---

## 七、迁移学习

```
思想：用预训练模型，微调最后几层

好处：
✓ 无需大量数据
✓ 训练更快
✓ 性能更好
```

### 使用预训练模型

```python
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models

# 1. 加载预训练模型 (不包括顶层)
base_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# 2. 冻结基础模型
base_model.trainable = False

# 3. 添加新层
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 你的分类任务
])

# 4. 编译和训练
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# 5. 微调 (解冻部分层)
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False  # 保持前面层冻结

model.compile(optimizer=keras.optimizers.Adam(1e-5),  # 更小学习率
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.1)
```

---

## 📝 本章小结

```
┌─────────────────────────────────────────────────────────────┐
│                     核心要点                                │
├─────────────────────────────────────────────────────────────┤
│ • CNN 优势：局部连接、权值共享、空间层次                    │
│ • 卷积层：提取特征 (边缘、纹理、形状...)                    │
│ • 池化层：降维、扩大感受野、平移不变                        │
│ • 经典架构：LeNet → AlexNet → VGG → ResNet                 │
│ • 数据增强：扩充数据，防止过拟合                            │
│ • 迁移学习：用预训练模型，快速适配新任务                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔜 下一章

[→ 4. 实战：图像分类项目](./04-图像分类实战.md)

---

*📊 图解机器学习系列 | 深度学习 第 3 章*
