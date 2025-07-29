import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from keras.models import load_model
from keras import Model
import copy
from DATIS.DATIS import DATIS_test_input_selection,DATIS_redundancy_elimination
from keras.datasets import mnist
import os
import numpy as np
import pandas as pd
import random
import scipy.io
from collections import Counter

import keras
import tensorflow as tf
from keras.models import load_model
from keras import Model
from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV

from scipy.stats import entropy

from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kendalltau, spearmanr
import numpy as np
import pandas as pd
from scipy import stats

def get_faults(sample, mis_ind_test, Clustering_labels):
    i=0
    pos=0
    neg=0
    i=0
    cluster_lab=[]
    nn=-1
    for l in sample:
        if l in mis_ind_test:
            neg=neg+1 
            ind=list(mis_ind_test).index(l)
            if (Clustering_labels[ind]>-1):
                cluster_lab.append(Clustering_labels[ind])
            if (Clustering_labels[ind]==-1):
                cluster_lab.append(nn)
                nn=nn-1
        else:
            pos=pos+1

    faults_n=len(list(set(cluster_lab)))
    #All noisy mispredicted inputs are considered as one specific fault

    cluster_1noisy=copy.deepcopy(cluster_lab)
    for i in range(len(cluster_1noisy)):
        if cluster_1noisy[i] <=-1:
            cluster_1noisy[i]=-1
    faults_1noisy=len(list(set(cluster_1noisy)))
    return faults_n,faults_1noisy, neg


def calculate_rate(budget_ratio_list,test_support_output,x_test,rank_lst,ans,cluster_path):


    top_list =[]
    for ratio_ in budget_ratio_list:
        top_list.append(int(len(x_test)*ratio_))
    result_fault_rate= []
    clustering_labels = np.load(cluster_path+'/cluster1.npy')
    fault_sum_all = np.max(clustering_labels)+1+np.count_nonzero(clustering_labels == -1)
    mis_test_ind= np.load(cluster_path+'/mis_test_ind.npy')


    print('total test case:{len}')

    for i_, n in enumerate(top_list):

        if len(ans)!=0:
            n_indices =ans[i_]
        else :
            n_indices = rank_lst[:n]

        n_fault,n_noisy,n_neg = get_faults(n_indices,mis_test_ind,clustering_labels)

        faults_rate = n_fault/min(n,fault_sum_all)

        print(f"The Fault Detection Rate of Top: {n} cases :{faults_rate}")
        result_fault_rate.append(faults_rate)

    return result_fault_rate



def load_data():  
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
      
        x_test = x_test.astype('float32')
        x_train = x_train.astype('float32')
        x_train /= 255
        x_test /= 255
        return (x_train, y_train), (x_test, y_test)


def load_data_corrupted():

    data_corrupted_file = "./corrupted_data/mnist/data_corrupted.npy"
    label_corrupted_file = "./corrupted_data/mnist/label_corrupted.npy"
    data_corrupted_file = "./corrupted_data/mnist/data_corrupted.npy"
    label_corrupted_file = "./corrupted_data/mnist/label_corrupted.npy"
    x_test_ood = np.load(data_corrupted_file)
    y_test_ood = np.load(label_corrupted_file)
    y_test_ood = y_test_ood.reshape(-1)
    x_test_ood = x_test_ood.reshape(-1, 28, 28, 1)
    x_test_ood = x_test_ood.astype('float32')
    x_test_ood /= 255
    return x_test_ood,y_test_ood
   
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


def load_data_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)

    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def load_data_fashion():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)

    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


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
    elif data_type == 'cifar100_corrupted.csv':
        (x_train, y_train), (x_test, y_test) = load_data_cifar100()
        x_test, y_test = load_data_corrupted()
        cluster_path = './cluster_data/ResNet32_cifar100_corrupted'
        model_path = "./model/model_cifar100_resNet32.h5"
    elif data_type == 'svhn_vgg16.csv':
        (x_train, y_train), (x_test, y_test) = load_svhn_data()
        cluster_path = './cluster_data/model_svhn_vgg16'
        model_path = "./model/model_svhn_vgg16.hdf5"
    elif data_type == 'svhn_lenet5.csv':
        (x_train, y_train), (x_test, y_test) = load_svhn_data()
        cluster_path = './cluster_data/model_svhn_LeNet5'
        model_path = "./model/model_svhn_LeNet5.hdf5"
    elif data_type == 'mnist_LeNet5.csv':
        (x_train, y_train), (x_test, y_test) = load_data_mnist()
        cluster_path = './cluster_data/LeNet5_mnist_nominal'
        model_path = "./model/model_mnist_LeNet5.hdf5"
    elif data_type == 'fashion_LeNet1.csv':
        (x_train, y_train), (x_test, y_test) = load_data_fashion()
        cluster_path = './cluster_data/fashion_mnist_lenet1'
        model_path = "./model/model_fashion_LeNet1.hdf5"
    elif data_type == 'fashion_resnet20.csv':
        (x_train, y_train), (x_test, y_test) = load_data_fashion()
        cluster_path = './cluster_data/fashion_mnist_resnet20'
        model_path = "./model/model_fashion_resNet20.hdf5"

    ori_model = load_model(model_path)
     
    
    new_model = Model(ori_model.input, outputs=ori_model.layers[-2].output)
    train_support_output = new_model.predict(x_train)
    train_support_output= np.squeeze(train_support_output)
    test_support_output=new_model.predict(x_test) 
    test_support_output= np.squeeze(test_support_output)
    softmax_test_prob = ori_model.predict(x_test)
        
    rank_lst = DATIS_test_input_selection(softmax_test_prob,train_support_output,y_train,test_support_output,y_test,10)
    

    budget_ratio_list =[0.001,0.005,0.01,0.02,0.03,0.05,0.1]

    ans= DATIS_redundancy_elimination(budget_ratio_list,rank_lst,test_support_output,y_test)

    results=calculate_rate(budget_ratio_list,test_support_output,x_test,rank_lst,ans,cluster_path)
    save_results_to_csv(results, data_type)
    
    

if __name__ == '__main__':
    datasets = [
                'fashion_LeNet1.csv']

    for dataset in datasets:
        print(f"Processing {dataset}")
        demo(dataset)
        print("=====================================")



   
