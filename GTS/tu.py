import matplotlib.pyplot as plt

# 数据
x = ['gnn5', 'gnn10', 'gnn20', 'gnn50', 'gnn70', 'gnn100', 'gnn200']
data = {
    "fashion-LeNet1": [0.560113105, 0.560113105, 0.560113105, 0.560113105, 0.560113105, 0.560113105, 0.560113105],
    "cifar10-resnet20": [0.537620381, 0.537989863, 0.540501534, 0.535658263, 0.53661598, 0.545211418, 0.535266106],
    "svhn-LeNet5": [0.498627143, 0.499137091, 0.499137091, 0.498313171, 0.499137091, 0.499137091, 0.499137091],
    "svhn-vgg16": [0.329954039, 0.329954039, 0.329954039, 0.329954039, 0.329954039, 0.329954039, 0.329954039],
    "fashion-resnet20": [0.715014577, 0.713702624, 0.716472303, 0.715014577, 0.715014577, 0.715014577, 0.71574344],
    "cifar10-vgg16": [0.323570762, 0.323570762, 0.323570762, 0.323570762, 0.323570762, 0.323570762, 0.323570762],
    "cifar100-resnet32-corrupted": [0.736621444, 0.734012137, 0.734012137, 0.734997062, 0.729632653, 0.732214131, 0.732387755],
    "cifar100-resnet32": [0.718316293, 0.723636751, 0.724004098, 0.726806819, 0.725615126, 0.712140883, 0.724153514],
}

# 绘制折线图
plt.figure(figsize=(10, 6))
for label, values in data.items():
    plt.plot(x, values, marker='o', label=label)

# 添加图例、标题和轴标签
plt.title('FDR Values for Different Models (GNN Data)', fontsize=14)
plt.xlabel('K_NGraph', fontsize=12)
plt.ylabel('FDR', fontsize=12)
plt.legend(title='Models', fontsize=10, loc='lower left', bbox_to_anchor=(0.0, 0.0))
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.show()
