import matplotlib.pyplot as plt

# 数据
x = ['ep10', 'ep20', 'ep30', 'ep50', 'ep100', 'ep200']
data = {

        "cifar10-resnet20": [0.484995331, 0.468347339, 0.468347339, 0.468347339, 0.484995331, 0.468347339],
        "cifar10-vgg16": [0.300653595, 0.300653595, 0.300653595, 0.300653595, 0.300653595, 0.300653595],
        "cifar100-resnet32": [0.594600776, 0.594600776, 0.594600776, 0.635461327, 0.635461327, 0.635461327],
        "fashion-mnist-LeNet1": [0.513012048, 0.513012048, 0.513012048, 0.513012048, 0.513012048, 0.513012048],
        "fashion-mnist-resnet20": [0.723469388, 0.728571429, 0.728571429, 0.728571429, 0.728571429, 0.728571429],
        "svhn-LeNet5": [0.478373914, 0.474854557, 0.474854557, 0.474854557, 0.474854557, 0.474854557],
        "svhn-vgg16": [0.310737455, 0.310737455, 0.310737455, 0.310737455, 0.310737455, 0.310737455]
}

# 绘制折线图
plt.figure(figsize=(10, 6))
for label, values in data.items():
    plt.plot(x, values, marker='o', label=label)

# 添加图例、标题和轴标签
plt.title('FDR Values for Different Models (GNN Data)', fontsize=14)
plt.xlabel('GNN epoch', fontsize=12)
plt.ylabel('FDR', fontsize=12)
plt.legend(title='Models', fontsize=10, loc='lower left', bbox_to_anchor=(0.0, 0.0))
plt.grid(True)
plt.tight_layout()

# 显示图表
plt.show()
