import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras
import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras import Model
from DATIS.DATIS import DATIS_test_input_selection,DATIS_redundancy_elimination
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
tf.random.set_seed(45)
epoch = {
    "nominal": 5,
    "corrupted": 19
}


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
   

     
def retrain(data_type,model_path, x_s, y_s, X_train, Y_train, x_val, y_val,nb_classes,
                   verbose=1):
   
    Ya_train = np.concatenate([Y_train,y_s])
    Xa_train = np.concatenate([X_train,x_s])
    
    Ya_train_vec = keras.utils.np_utils.to_categorical(Ya_train, nb_classes)

    ori_model = load_model(model_path)

    Y_val_vec = keras.utils.np_utils.to_categorical(y_val, nb_classes)
    
    acc_base_val = ori_model.evaluate(x_val, Y_val_vec, verbose=0)[1]
    
    trained_model = ori_model

    trained_model.fit(Xa_train, Ya_train_vec, batch_size=128, epochs=epoch[data_type],validation_data=(x_val, Y_val_vec),verbose=1)
    
    acc_si_val = trained_model.evaluate(x_val, Y_val_vec, verbose=0)[1]

    acc_imp_val = acc_si_val - acc_base_val

    print("val acc", acc_base_val, acc_si_val, "improvement:", format(acc_imp_val, ".4f"))
    
    return



    
def demo(data_type):
   
    if data_type =='nominal':
        (x_train, y_train), (x_test, y_test) = load_data()
        cluster_path ='./cluster_data/LeNet5_mnist_nominal'

    elif data_type == 'corrupted':
        (x_train, y_train), (x_test, y_test)= load_data()
        x_test, y_test = load_data_corrupted()
        cluster_path ='./cluster_data/LeNet5_mnist_corrupted'
    
    data_length =len(x_test)
    mid_index = data_length // 2
    x_test_ori = x_test
    y_test_ori = y_test

    x_test = x_test_ori[:mid_index]
    y_test = y_test_ori[:mid_index]

    x_val = x_test_ori[mid_index:]
    y_val = y_test_ori[mid_index:]

    model_path = "./model/model_mnist_LeNet5.hdf5"
   
    ori_model = load_model(model_path)
     
    
    new_model = Model(ori_model.input, outputs=ori_model.layers[-2].output)
    train_support_output = new_model.predict(x_train)
    train_support_output= np.squeeze(train_support_output)
    test_support_output=new_model.predict(x_test) 
    test_support_output= np.squeeze(test_support_output)
    softmax_test_prob = ori_model.predict(x_test)
       
        
    rank_lst =DATIS_test_input_selection(softmax_test_prob,train_support_output,y_train,test_support_output,y_test,10)
    
    
    nb_classes =10

    #select top 10%
    budget_ratio_list =[0.1]

    ans= DATIS_redundancy_elimination(budget_ratio_list,rank_lst,test_support_output,y_test)
    
    x_s, y_s = x_test[ans[0]], y_test[ans[0]]

    retrain(data_type,model_path, x_s, y_s, x_train, y_train,x_val, y_val,nb_classes,verbose=0)

    
    

if __name__ == '__main__':
   
    demo('nominal')
    print("         =====================================           ")
    demo('corrupted')
    

   
