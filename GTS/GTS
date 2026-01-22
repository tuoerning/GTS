import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
#备份使用

# 计算每个样本与其 K 个最近邻样本的 SSIM 指数
def calculate_ssim_k_neighbors(data, k=100):
    """
    计算每个样本与其K个最近邻样本之间的SSIM指数。

    参数:
        - data: 数据集 (numpy array)，每行是一个样本的特征
        - k: 最近邻数量

    返回:
        - ssim_matrix: 每个样本与其K个最近邻之间的SSIM值矩阵
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)  # 用 K 个邻居拟合 NearestNeighbors 模型
    _, indices = nbrs.kneighbors(data)  # 获取每个样本的 K 个邻居的索引

    num_samples = len(data)  # 获取样本数量
    ssim_matrix = np.zeros((num_samples, k))  # 初始化 SSIM 值矩阵

    for i in range(num_samples):  # 遍历每个样本
        for j in range(1, k):  # 从 1 开始，跳过自身
            ssim_index = ssim(data[i], data[indices[i][j]], data_range=1.0)  # 计算 SSIM 指数
            ssim_matrix[i][j] = ssim_index  # 存储 SSIM 值到矩阵中

    return ssim_matrix  # 返回 SSIM 矩阵


# 测试输入选择函数
def DATIS_test_input_selection(softmax_prob, train_support_output, y_train, test_support_output, y_test, num_classes,
                               k=100, T=0.1):
    # 使用 L2 范数归一化
    normalizer = Normalizer(norm='l2')
    train_support_output = normalizer.transform(train_support_output)
    test_support_output = normalizer.transform(test_support_output)

    # 初始化 KNN 分类器并训练
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_support_output, y_train)

    metrics = []
    prob_test = np.zeros((len(test_support_output), num_classes))

    # 计算测试样本的 KNN 相似性与 SSIM 相似性
    ssim_matrix = calculate_ssim_k_neighbors(test_support_output, k=k)

    for i, z in tqdm(enumerate(test_support_output), total=len(test_support_output)):
        z = z.reshape(1, -1) if z.ndim == 1 else z
        distance, indices = knn.kneighbors(z.reshape(1, -1), n_neighbors=k)
        support_points = train_support_output[indices.flatten()]
        support_labels = y_train[indices.flatten()]
        support_labels = support_labels.flatten()

        # 计算每个邻居的加权距离 (KNN)
        distance_sum = -np.sum((z - support_points) ** 2, axis=1) / T
        denominator = np.sum(np.exp(distance_sum))

        # 计算每个类别的概率
        for j in range(num_classes):
            num_rator = np.multiply(np.exp(distance_sum), (support_labels == j))
            numerator = np.sum(num_rator)
            prob_test[i][j] = numerator / denominator

        # 投票机制
        label_counts = np.bincount(support_labels, minlength=num_classes)
        voting_score = np.zeros(num_classes)
        for j in range(num_classes):
            voting_score[j] = label_counts[j] / k
        prob_test[i] += voting_score

        # SSIM 相似性权重计算
        ssim_weights = np.mean(ssim_matrix[i, 1:])  # 忽略自身
        prob_test[i] *= ssim_weights  # 将 SSIM 权重应用到概率上

    # 获取 softmax 和最终概率的最大索引
    softmax_max_indices = np.argmax(softmax_prob, axis=1)
    max_indices = np.argmax(prob_test, axis=1)
    temp = prob_test.copy()
    for i in range(len(max_indices)):
        temp[i][max_indices[i]] = -1
    second_max_indices = np.argmax(temp, axis=1)

    metrics = []
    epsilon = 1e-15
    for i in range(len(max_indices)):
        # 判断当前样本的最大概率类别（max_indices[i]）是否与模型预测的类别（softmax_max_indices[i]）一致
        if max_indices[i] == softmax_max_indices[i]:
            # 如果一致，计算第二大概率与最大概率的比值
            a = prob_test[i][second_max_indices[i]]
            b = prob_test[i][softmax_max_indices[i]]
        else:
            # 如果不一致，计算当前最大概率与模型预测类别的比值
            a = prob_test[i][max_indices[i]]
            b = prob_test[i][softmax_max_indices[i]]

        # 比较最大概率类别与模型预测结果
        if max_indices[i] != softmax_max_indices[i]:
            # 如果最大概率类别与模型预测类别不一致，增加排序权重
            # 使用一个惩罚因子（例如10），确保该样本排在前面
            penalty_factor = 10  # 你可以根据需求调整这个因子
            metrics.append((a / (b + epsilon)) * penalty_factor)  # 乘以惩罚因子增加排序优先级
        else:
            # 如果一致，则按原有的概率比值进行排序
            metrics.append(a / (b + epsilon))  # 保持原有排序，即按概率比值排序

    # 排序并返回索引
    rank_lst = np.argsort(metrics)[::-1]
    return rank_lst


# 冗余特征消除
def DATIS_redundancy_elimination(budget_ratio_list, rank_list, test_support_output, y_pred, k=5, noise_threshold=0.01):
    size = len(test_support_output)
    normalizer = Normalizer(norm='l2')
    test = normalizer.transform(test_support_output)
    ratio_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    pool_list = [4, 3, 3, 2, 2, 2, 2]
    weight_list = [0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]

    top_list = []
    arg_index_list = []
    for ratio_ in budget_ratio_list:
        top_list.append(int(size * ratio_))
        for i, ratio in enumerate(ratio_list):
            if ratio_ == ratio:
                arg_index_list.append(i)

    ans = []
    for i_, k in tqdm(enumerate(top_list), total=len(top_list)):
        index = arg_index_list[i_]
        tmp_k = int(k * pool_list[index])

        # Ensure tmp_k does not exceed the length of rank_list
        tmp_k = min(tmp_k, len(rank_list))
        selected_indices = rank_list[:tmp_k]

        tmp_set = test[selected_indices, :]
        tmp_label = y_pred[selected_indices]

        kn = min(100, k)
        knn = KNeighborsClassifier(n_neighbors=kn)
        knn.fit(tmp_set, tmp_label)

        distances = np.zeros(tmp_k)
        for i, z in enumerate(tmp_set):
            distance, indices = knn.kneighbors(z.reshape(1, -1), n_neighbors=kn)
            distances[i] = np.mean(distance)

        ssim_matrix = calculate_ssim_k_neighbors(tmp_set, k)
        average_ssim = np.mean(ssim_matrix[:, 1:], axis=1)

        # Filter selected_indices based on noise threshold
        selected_indices = selected_indices[average_ssim >= noise_threshold]

        # Recalculate weights with the filtered selected_indices
        step1_weights = np.arange(len(selected_indices), 0, -1)
        rank_weights = np.argsort(-distances)[:len(selected_indices)]
        distance_weights = np.zeros(len(selected_indices))
        weight_k = len(selected_indices)

        for i in rank_weights:
            distance_weights[i] = weight_k
            weight_k -= 1

        weights = (1 - weight_list[index]) * step1_weights + weight_list[index] * distance_weights
        sorted_indices = selected_indices[np.argsort(-weights)][:int(k)]
        ans.append(sorted_indices)

    return ans
