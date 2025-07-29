import matplotlib.pyplot as plt

# 数据
x = ['gts5', 'gts10', 'gts20', 'gts50', 'gts70', 'gts100', 'gts200']
data = {
    "fashion-LeNet1": [0.513012048, 0.557693632, 0.567401033, 0.6, 0.565129088, 0.562822719, 0.554733219],
    "cifar10-resnet20": [0.476631186, 0.510436975, 0.516095238, 0.51831746, 0.524423903, 0.512724556, 0.508065359],
    "svhn-LeNet5": [0.474854557, 0.49457937, 0.502197802, 0.502046973, 0.503641457, 0.507598937, 0.507706672],
    "svhn-vgg16": [0.310737455, 0.315900967, 0.319343307, 0.338607176, 0.342380511, 0.344101681, 0.338607176],
    "fashion-resnet20": [0.728571429, 0.709183673, 0.726239067, 0.738483965, 0.703061224, 0.696355685, 0.704081633],
    "cifar10-vgg16": [0.300653595, 0.318506069, 0.349112979, 0.32774043, 0.324220355, 0.325210084, 0.319551821],
    "cifar100-resnet32-corrupted": [0.654325448, 0.7450797, 0.74656107, 0.75132684, 0.754981061, 0.741457947, 0.740145254],
    "cifar100-resnet32": [0.623786884, 0.733438241, 0.741428544, 0.753699423, 0.756140396, 0.734723739, 0.711456256],
}

# 绘制折线图
plt.figure(figsize=(10, 6))
for label, values in data.items():
    plt.plot(x, values, marker='o', label=label)

# 添加图例、标题和轴标签
plt.title('FDR Values for Different Models', fontsize=14)
plt.xlabel('K_TSort ', fontsize=12)
plt.ylabel('FDR', fontsize=12)
plt.legend(title='Models', fontsize=10, loc='lower left', bbox_to_anchor=(0.0, 0.0))
plt.grid(True)
plt.tight_layout()
# 显示图表
plt.show()
