from GTS import GTS_test_input_selection, GTS_redundancy_elimination
import os
import numpy as np
from keras.models import load_model
from keras import Model
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from keras.datasets import cifar10, cifar100
import scipy.io
import random
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv, GATConv
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 设置环境变量，禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# 定义一个简单的图神经网络（GNN）模型类
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = torch.nn.Linear(input_dim, 16)  # 第一个图卷积层，输出为 16 维特征
        self.conv2 = torch.nn.Linear(16, output_dim)  # 第二个图卷积层，输出为最终的 output_dim 维度

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x))  # 通过第一个图卷积层并应用 ReLU 激活函数
        x = self.conv2(x)  # 通过第二个图卷积层
        return x


# 随机删除一定比例的特征，将其置零而不改变形状
def random_remove_features(features, removal_ratio=0.1):
    num_features = features.shape[1]
    num_to_remove = int(num_features * removal_ratio)
    to_remove_indices = random.sample(range(num_features), num_to_remove)  # 随机选择要置零的特征索引

    features_copy = features.copy()  # 创建特征的副本，以免修改原始数据
    for col in to_remove_indices:
        features_copy[:, col] = 0  # 将选中的特征列置零

    print(f"Randomly zeroed out {len(to_remove_indices)} features out of {num_features}")
    return features_copy


# 构建 K 近邻图，将特征转为邻接矩阵
def get_knn_graph(features, n_neighbors=10):
    knn = NearestNeighbors(n_neighbors=n_neighbors)  # 使用 n_neighbors 指定邻居数量的 KNN 模型
    knn.fit(features)  # 拟合特征数据
    adjacency_matrix = knn.kneighbors_graph(features, mode='connectivity').toarray()  # 转为邻接矩阵
    return adjacency_matrix


# 基于预测不确定性生成置信度标签
def generate_confidence_labels(softmax_probabilities, threshold=0.5):
    # 计算每个样本的不确定性（熵）
    uncertainties = -np.sum(softmax_probabilities * np.log(softmax_probabilities + 1e-10), axis=1)
    # 将不确定性转换为置信度标签（低不确定性对应高置信度）
    confidence_labels = (uncertainties < threshold).astype(int)
    return confidence_labels


# 训练图神经网络 (GNN) 并根据置信度区分特征
def train_gnn(features, adjacency_matrix, confidence_labels):
    edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long)  # 将邻接矩阵转换为边索引格式
    x = torch.tensor(features, dtype=torch.float)  # 将输入特征转换为张量
    data = Data(x=x, edge_index=edge_index)  # 创建图数据，包含特征和边索引
    model = GNNModel(input_dim=features.shape[1], output_dim=features.shape[1])  # 初始化 GNN 模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义 Adam 优化器，学习率为 0.01

    model.train()  # 设置模型为训练模式
    for epoch in range(100):  # 训练 100 个 epoch
        optimizer.zero_grad()  # 清除梯度
        out = model(data.x, data.edge_index)  # 进行前向传播
        confidence_mask = torch.tensor(confidence_labels, dtype=torch.float).unsqueeze(1)  # 将置信度标签转换为张量
        weighted_out = out * confidence_mask  # 对输出特征应用置信度权重
        loss = F.mse_loss(weighted_out, data.x * confidence_mask)  # 计算加权后的均方误差损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数


# 移除高低置信度特征之间的相似特征
def construct_graph(features, threshold=0.8):
    sim_matrix = cosine_similarity(features.T)
    edge_index = np.array(np.nonzero(sim_matrix > threshold))
    edge_weight = sim_matrix[edge_index[0], edge_index[1]]
    edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
    return Data(
        x=torch.tensor(features.T, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
    )


class GNNFeatureSelector(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNFeatureSelector, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


def remove_redundant_features_with_gnn(features, confidence_labels, similarity_threshold=0.5, num_epochs=50,
                                       base_learning_rate=0.001):
    """
    使用 GNN 模型移除冗余特征。
    """

    # 特征标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 构造图
    graph_data = construct_graph(features, threshold=similarity_threshold)

    # 减小隐藏层大小
    hidden_channels = 16

    # 初始化 GNN 模型
    gnn = GNNFeatureSelector(in_channels=graph_data.x.size(1), hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=base_learning_rate)

    # 训练 GNN
    gnn.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        scores = gnn(graph_data)

        # 检查分数是否有效
        if torch.isnan(scores).any():
            print(f"Epoch {epoch}: Detected NaN in scores! Skipping epoch...")
            continue

        # 匹配高低置信度样本
        high_mask = confidence_labels == 1
        low_mask = confidence_labels == 0
        high_features = features[high_mask]
        low_features = features[low_mask]

        high_confidence_scores = scores[:high_features.shape[0]]
        low_confidence_scores = scores[high_features.shape[0]:]

        # 简化损失函数
        high_loss = F.mse_loss(high_confidence_scores, torch.ones_like(high_confidence_scores))
        low_loss = F.mse_loss(low_confidence_scores, torch.zeros_like(low_confidence_scores))

        # 总损失
        loss = high_loss + low_loss
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # 反向传播
        loss.backward()
        optimizer.step()

    # 推理阶段
    gnn.eval()
    with torch.no_grad():
        scores = gnn(graph_data).numpy()
        print(f"Scores: {scores}")
        threshold = np.percentile(scores, 10)  # 选取最低 10% 的得分
        to_remove = np.where(scores < threshold)[0]

    print(f"Number of features to zero out: {len(to_remove)}")
    print(f"Original features shape: {features.shape}")

    if len(to_remove) > 0:
        features[:, to_remove] = 0

    print(f"Features shape after zeroing out: {features.shape}")
    return features


# 加载 CIFAR-100 数据集
def load_data_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # 加载 CIFAR-10 训练和测试数据
    x_test = x_test.reshape(-1, 32, 32, 3)  # 重塑测试集形状
    x_train = x_train.reshape(-1, 32, 32, 3)  # 重塑训练集形状
    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')
    x_train /= 255  # 归一化训练集
    x_test /= 255  # 归一化测试集

    # 将标签形状统一为 (num_samples,)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return (x_train, y_train), (x_test, y_test)  # 返回数据集


def load_data_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()  # 加载 CIFAR-10 训练和测试数据
    x_test = x_test.reshape(-1, 32, 32, 3)  # 重塑测试集形状
    x_train = x_train.reshape(-1, 32, 32, 3)  # 重塑训练集形状
    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')
    x_train /= 255  # 归一化训练集
    x_test /= 255  # 归一化测试集

    # 将标签形状统一为 (num_samples,)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return (x_train, y_train), (x_test, y_test)  # 返回数据集


# 加载被干扰的 CIFAR-100 数据集
def load_data_corrupted():
    data_corrupted_file = "./corrupted_data/cifar100/data_corrupted.npy"  # 干扰数据文件路径
    label_corrupted_file = "./corrupted_data/cifar100/label_corrupted.npy"  # 干扰标签文件路径
    x_test_ood = np.load(data_corrupted_file)  # 加载干扰测试数据
    y_test_ood = np.load(label_corrupted_file)  # 加载干扰标签
    y_test_ood = y_test_ood.reshape(-1)
    x_test_ood = x_test_ood.reshape(-1, 32, 32, 3)
    x_test_ood = x_test_ood.astype('float32')
    x_test_ood /= 255  # 归一化干扰数据
    return x_test_ood, y_test_ood


# 计算检测率，评估测试集选择的效果
def calculate_rate(budget_ratio_list, test_support_output, x_test, rank_lst, ans, cluster_path, random=False):
    top_list = [int(len(x_test) * ratio) for ratio in budget_ratio_list]  # 根据比例计算要测试的用例数量
    result_fault_rate = []  # 存储每个比例下的错误检测率
    clustering_labels = np.load(cluster_path + '/cluster1.npy')  # 加载聚类标签
    fault_sum_all = np.max(clustering_labels) + 1 + np.count_nonzero(clustering_labels == -1)  # 计算所有的故障数量
    mis_test_ind = np.load(cluster_path + '/mis_test_ind.npy')  # 加载错分样本索引

    for i_, n in enumerate(top_list):  # 遍历每种预算比例
        if random:
            np.random.shuffle(rank_lst)  # 如果 random 为 True，打乱顺序
        n_indices = rank_lst[:n] if not random else np.random.choice(len(x_test), n, replace=False)  # 获取前 n 个索引
        n_fault, n_noisy, n_neg = get_faults(n_indices, mis_test_ind, clustering_labels)  # 计算故障率
        faults_rate = n_fault / min(n, fault_sum_all)  # 计算故障检测率
        print(f"The {'Random ' if random else ''}Fault Detection Rate of Top {n} cases: {faults_rate}")
        result_fault_rate.append(faults_rate)

    return result_fault_rate  # 返回完整的错误检测率列表


# 根据样本索引计算故障情况
def get_faults(sample, mis_ind_test, clustering_labels):
    pos, neg = 0, 0  # 初始化正例和负例计数
    nn = -1  # 初始化噪声标记
    cluster_lab = []  # 初始化样本的聚类标签列表
    for l in sample:
        if l in mis_ind_test:
            neg += 1  # 如果样本是错分样本，增加负例计数
            ind = list(mis_ind_test).index(l)
            cluster_lab.append(clustering_labels[ind] if clustering_labels[ind] > -1 else nn)
            if clustering_labels[ind] == -1:
                nn -= 1
        else:
            pos += 1  # 否则，增加正例计数

    faults_n = len(set(cluster_lab))  # 计算不同的故障数量
    cluster_1noisy = [i if i > -1 else -1 for i in cluster_lab]  # 处理噪声标签
    faults_1noisy = len(set(cluster_1noisy))  # 计算非噪声故障数量
    return faults_n, faults_1noisy, neg


def load_svhn_data():
    # 从.mat文件加载SVHN数据集
    train_data = scipy.io.loadmat('train_32x32.mat')
    test_data = scipy.io.loadmat('test_32x32.mat')

    x_train = train_data['X'].transpose((3, 0, 1, 2))
    y_train = train_data['y'].flatten()
    x_test = test_data['X'].transpose((3, 0, 1, 2))
    y_test = test_data['y'].flatten()

    x_train = x_train.astype('float32') / 255  # 归一化
    x_test = x_test.astype('float32') / 255  # 归一化

    y_train = np.where(y_train == 10, 0, y_train)  # 将标签10转换为0
    y_test = np.where(y_test == 10, 0, y_test)  # 将标签10转换为0

    return (x_train, y_train), (x_test, y_test)


def save_results_to_csv(results, filename="experiment_results.csv"):
    # 将字典列表转换为 DataFrame
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def calculate_deepgini_scores(softmax_probabilities):
    """
    Calculate DeepGini scores for a given set of softmax probabilities.
    Lower DeepGini score indicates higher uncertainty.
    """
    squared_probs = np.sum(np.square(softmax_probabilities), axis=1)
    deepgini_scores = 1 - squared_probs  # DeepGini score calculation
    return deepgini_scores


def deepgini_test_selection(softmax_probabilities, budget_ratios, x_test, y_test, cluster_path):
    """
    Perform test case selection based on DeepGini scores and calculate fault detection rate.
    """
    deepgini_scores = calculate_deepgini_scores(softmax_probabilities)
    rank_indices = np.argsort(-deepgini_scores)  # Rank by ascending DeepGini scores (low to high uncertainty)

    print("Results based on DeepGini value ranking:")
    return calculate_rate(budget_ratios, softmax_probabilities, x_test, rank_indices, y_test, cluster_path)
def sort_and_test_cases(model, x_test, y_test, budget_ratio_list, cluster_path):
    # Predict probabilities for test set and get max probability per test case
    softmax_prob = model.predict(x_test)
    maxp_values = softmax_prob.max(axis=1)

    # Rank test cases by descending order of maxp
    maxp_rank_indices = np.argsort(maxp_values)

    print("Results based on Maxp value ranking:")
    a = calculate_rate(budget_ratio_list, softmax_prob, x_test, maxp_rank_indices, y_test, cluster_path)
    return a

def retrain(data_type, model_path, x_s, y_s, X_train, Y_train, x_val, y_val, nb_classes,
            verbose=1):
    Ya_train = np.concatenate([Y_train, y_s])
    Xa_train = np.concatenate([X_train, x_s])

    Ya_train_vec = keras.utils.np_utils.to_categorical(Ya_train, nb_classes)

    ori_model = load_model(model_path)

    Y_val_vec = keras.utils.np_utils.to_categorical(y_val, nb_classes)

    acc_base_val = ori_model.evaluate(x_val, Y_val_vec, verbose=0)[1]

    trained_model = ori_model

    trained_model.fit(Xa_train, Ya_train_vec, batch_size=128, epochs=epoch[data_type],
                      validation_data=(x_val, Y_val_vec), verbose=1)

    acc_si_val = trained_model.evaluate(x_val, Y_val_vec, verbose=0)[1]

    acc_imp_val = acc_si_val - acc_base_val

    print("val acc", acc_base_val, acc_si_val, "improvement:", format(acc_imp_val, ".4f"))

    return
# 主函数，执行实验方案和随机选择对比实验
def demo(data_type):
    if data_type == 'cifar10_vgg16.csv':
        (x_train, y_train), (x_test, y_test) = load_data_cifar10()
        cluster_path = './cluster_data/cifar_vgg16'
        model_path = "./model/model_cifar_vgg16.hdf5"
    elif data_type == 'cifar10_resnet20.csv':
        (x_train, y_train), (x_test, y_test) = load_data_cifar10()
        cluster_path = './cluster_data/cifar_resnet20'
        model_path = "./model/model_cifar_resNet20.h5"
    elif data_type == 'cifar100.csv':
        (x_train, y_train), (x_test, y_test) = load_data_cifar100()
        cluster_path = './cluster_data/ResNet32_cifar100_nominal'
        model_path = "./model/model_cifar100_resNet32.h5"
    elif data_type == 'svhn_vgg16.csv':
        (x_train, y_train), (x_test, y_test) = load_svhn_data()
        cluster_path = './cluster_data/model_svhn_vgg16'
        model_path = "./model/model_svhn_vgg16.hdf5"
    elif data_type == 'svhn_lenet5.csv':
        (x_train, y_train), (x_test, y_test) = load_svhn_data()
        cluster_path = './cluster_data/model_svhn_LeNet5'
        model_path = "./model/model_svhn_LeNet5.hdf5"

    ori_model = load_model(model_path)
    new_model = Model(ori_model.input, outputs=ori_model.layers[-2].output)
    train_support_output = np.squeeze(new_model.predict(x_train))
    test_support_output = np.squeeze(new_model.predict(x_test))

    # 使用基于不确定性生成的置信度标签（训练集）
    softmax_train_prob = ori_model.predict(x_train)
    train_confidence_labels = generate_confidence_labels(softmax_train_prob)

    # # 使用基于置信度的过滤特征（训练集）
    filtered_train_features = remove_redundant_features_with_gnn(train_support_output, train_confidence_labels)
    knn_graph = get_knn_graph(filtered_train_features)

    # 生成测试集的软标签并生成测试集的置信度标签
    softmax_test_prob = ori_model.predict(x_test)
    test_confidence_labels = generate_confidence_labels(softmax_test_prob)

    # 使用测试集的置信度标签过滤测试集特征
    filtered_test_features = remove_redundant_features_with_gnn(test_support_output, test_confidence_labels)

    rank_lst = GTS_test_input_selection(softmax_test_prob, train_support_output, y_train, test_support_output,
                                        y_test, 100)
    budget_ratio_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    ans = GTS_redundancy_elimination(budget_ratio_list, rank_lst, test_support_output, y_test)
    x_s, y_s = x_test[ans[0]], y_test[ans[0]]
    retrain(data_type, model_path, x_s, y_s, x_train, y_train, x_val, y_val, nb_classes, verbose=0)
    # 实验方案检测率
    print("Experimental Approach Results:")
    exp_results = calculate_rate(budget_ratio_list, filtered_test_features, x_test, rank_lst, ans, cluster_path)

    # 随机删除特征并计算检测率
    print("Random Feature Removal Approach Results:")
    random_features = random_remove_features(train_support_output, removal_ratio=0.1)  # 随机删除10%的特征
    random_knn_graph = get_knn_graph(random_features)
    random_rank_lst = DATIS_test_input_selection(softmax_test_prob, random_features, y_train, filtered_test_features,
                                                 y_test, 100)
    random_ans = DATIS_redundancy_elimination(budget_ratio_list, random_rank_lst, filtered_test_features, y_test)
    random_results = calculate_rate(budget_ratio_list, test_support_output, x_test, random_rank_lst, random_ans,
                                    cluster_path)
    print("DATIS Selection Results:")
    DATIS_rank_lst = DATIS_test_input_selection(softmax_test_prob, train_support_output, y_train, test_support_output,
                                                y_test, 10)

    DATIS_ans = DATIS_redundancy_elimination(budget_ratio_list, rank_lst, test_support_output, y_test)

    DATIS_results = calculate_rate(budget_ratio_list, test_support_output, x_test, DATIS_rank_lst, DATIS_ans,
                                   cluster_path, random=True)
    # 随机选择对比检测率
    print("Random Selection Results:")
    random_select_results = calculate_rate(budget_ratio_list, test_support_output, x_test, rank_lst, ans, cluster_path,
                                           random=True)
    # DeepGini 方法检测率
    deepgini_results = deepgini_test_selection(softmax_test_prob, budget_ratio_list, x_test, y_test, cluster_path)

    # 基于 softmax 概率进行排序并计算检测率
    print("Softmax Probability Sorting Results:")
    softmax_results = sort_and_test_cases(ori_model, x_test, y_test, budget_ratio_list, cluster_path)

    # 确保结果格式一致
    if isinstance(exp_results, (float, int)):
        exp_results = [exp_results]
    if isinstance(random_results, (float, int)):
        random_results = [random_results]
    if isinstance(random_select_results, (float, int)):
        random_select_results = [random_select_results]
    if isinstance(softmax_results, (float, int)):
        softmax_results = [softmax_results]
    if isinstance(DATIS_results, (float, int)):
        DATIS_results = [DATIS_results]

    # 结果汇总
    results = {
        "Experiment Type": ['Experimental Approach', 'Random Feature Removal', 'Random Selection', 'Softmax Sorting',
                            'DATIS_Selection','DeepGini'],
        "Budget Ratio 0.001": [exp_results[0], random_results[0], random_select_results[0], softmax_results[0],
                               DATIS_results[0],deepgini_results[0]],
        "Budget Ratio 0.005": [exp_results[1], random_results[1], random_select_results[1], softmax_results[1],
                               DATIS_results[1],deepgini_results[1]],
        "Budget Ratio 0.01": [exp_results[2], random_results[2], random_select_results[2], softmax_results[2],
                              DATIS_results[2],deepgini_results[2]],
        "Budget Ratio 0.02": [exp_results[3], random_results[3], random_select_results[3], softmax_results[3],
                              DATIS_results[3],deepgini_results[3]],
        "Budget Ratio 0.03": [exp_results[4], random_results[4], random_select_results[4], softmax_results[4],
                              DATIS_results[4],deepgini_results[4]],
        "Budget Ratio 0.05": [exp_results[5], random_results[5], random_select_results[5], softmax_results[5],
                              DATIS_results[5],deepgini_results[5]],
        "Budget Ratio 0.1": [exp_results[6], random_results[6], random_select_results[6], softmax_results[6],
                             DATIS_results[6],deepgini_results[6]]
    }

    save_results_to_csv(results, data_type)


if __name__ == '__main__':
    # demo('cifar10_vgg16.csv')  # 选择 CIFAR-10
    # demo('cifar10_resnet20.csv')
    print("         =====================================           ")
    demo('cifar100.csv')  # 选择 CIFAR-100
    print("         =====================================           ")
    # demo('svhn_vgg16.csv')  # 选择 SVHN
    # demo('svhn_lenet5.csv')
