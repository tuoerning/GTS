import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.family'] = ['SimHei']  # 中文字体，如 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 横轴标签
tests = ["T1", "T2", "T3", "T4"]

# 多样性数据
data = {
    "WD": {
        "ImageNet-BDTest": [11.31, 19.62, 22.05, 31.25],
        "ImageNet-A": [12.08, 19.32, 23.06, 28.89],
        "Random": [1.69, 5.86, 7.49, 7.02]
    },
    "EE": {
        "ImageNet-BDTest": [7.91, 8.63, 9.16, 9.82],
        "ImageNet-A": [7.63, 9.11, 8.29, 10.48],
        "Random": [7.24, 8.25, 8.68, 9.51]
    },
    "LD": {
        "ImageNet-BDTest": [971, 986, 1010, 1027],
        "ImageNet-A": [983, 987, 998, 1019],
        "Random": [951, 961, 971, 1002]
    },
    "DWCMS-L": {
        "ImageNet-BDTest": [1462, 1486, 1496, 1511],
        "ImageNet-A": [1477, 1480, 1486, 1491],
        "Random": [1460, 1465, 1476, 1481]
    },
    "DWCMS-E": {
        "ImageNet-BDTest": [7.68, 8.45, 9.42, 10.96],
        "ImageNet-A": [7.36, 8.45, 9.73, 10.67],
        "Random": [7.19, 7.63, 8.01, 9.71]
    }
}

# 绘图并保存
for metric, values_dict in data.items():
    plt.figure(figsize=(6, 4))
    for label, values in values_dict.items():
        plt.plot(tests, values, marker='o', label=label)
    plt.xlabel("测试集", fontsize=12)
    plt.ylabel("多样性分数", fontsize=12)
    plt.title(metric, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 保存图片
    plt.savefig(f"{metric}.png", dpi=300)
    plt.close()  # 不显示图
