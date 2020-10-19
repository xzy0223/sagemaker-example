import os
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, metrics, Sequential

# from hello import hello

# 设置tensorflow日志等级，0:无屏蔽，1:屏蔽info，2:屏蔽warning及以下，3:屏蔽error及以下
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
获取模型函数，通过keras layer构建一个普通的DNN
reture：karas model对象
'''
def get_model():
    
    # 使用keras的Sequential类，创建一个容器，其中定义每一层的Layer
    model = Sequential([
        layers.Dense(256, input_shape=(784,), activation=tf.nn.relu, # 增加input_shape参数，否则会有警告
                     kernel_regularizer=keras.regularizers.l2(0.01)),  # w:[784, 256], b:[256], 激活函数使用relu, 设置l2正则化抑制过拟合
        layers.Dense(128, activation=tf.nn.relu,
                     kernel_regularizer=keras.regularizers.l2(0.01)),  # w:[256, 128], b:[128], 激活函数使用relu
        layers.Dense(64, activation=tf.nn.relu,
                     kernel_regularizer=keras.regularizers.l2(0.01)),  # w:[128, 64], b:[64], 激活函数使用relu
        layers.Dense(32, activation=tf.nn.relu,
                     kernel_regularizer=keras.regularizers.l2(0.01)),  # w:[64, 32], b:[32], 激活函数使用relu
        layers.Dense(10, activation=tf.nn.softmax)  # w:[64, 10], b:[10], 为了输出一个one-hot概率值，这里使用softmax
    ])

    # build网络，指定输入数据的格式，这个输入的shape一定要跟真实数据一致
    model.build(input_shape=[None, 28*28])
    # 打印出网络的汇总信息，每层类型和参数等
    model.summary()
    
    return model


'''
训练算法，训练多个epochs，每100个batch输出一次loss，并且每个epoch输出一次准确率
paras：
    model：keras model对象
    train_db：keras dataset对象
    validation_db：keras dataset对象
    epoch
    batch_size
    learning_rate
return：None
'''
def train(model, train_db, validation_db, epochs, batch_size, learning_rate, model_dir):
    
    # 对dataset进行处理
    # 预处理传入process函数
    # shuffle用于把数据打散，参数越大打的越散
    # 设定每一个batch的大小
    train_db = train_db.map(process).shuffle(10000).batch(batch_size)
    validation_db = validation_db.map(process).shuffle(10000).batch(batch_size)

    # 获得sample数据，查看数据的shape信息，用于下一步定义网络的一些参数
    train_iter = iter(train_db)
    train_sample = next(train_iter)
    print('train dataset x shape {}, train dataset y shape {}'.format(train_sample[0].shape, train_sample[1].shape))

    # optimazer用于更新参数，优化loss，设定learning rate
    optimazer = optimizers.Adam(lr=learning_rate)
    
    # 设定epoch的次数
    for epoch in range(epochs):
        
        for step, (x, y) in enumerate(train_db):
            # 对input进行reshape，对应model build的input_shape
            # x = tf.reshape(x, [-1,28*28])，不需要reshape了，传进来的数据已经是[b, 784]
            # tape包裹前向运算，用于记录varibale，好计算梯度
            with tf.GradientTape() as tape:
                # 直接将x输入model，实际上调用的实例的__call__方法，输出结果为softmax后的预测数据
                softmax = model(x) 
                # 对y进行onehot编码，因为y的shape是[b,],而logits的shape是[b, 10],需要将y转换成[b, 10]
                y_hot = tf.one_hot(y, depth=10)
                # 这里设置两个损失函数mse和交叉熵，如果是分类问题，推荐使用交叉熵
                # 注意，如果是使用logits计算交叉熵的化，需要指定参数from_logits=True，这里是softmax所以不需要了
                loss_mse = tf.reduce_mean(tf.losses.mean_squared_error(y_hot, softmax))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_hot, softmax))
                # loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_hot, logits, from_logits=True))
            # 计算对于交叉熵的参数的梯度   
            grads = tape.gradient(loss_ce, model.trainable_variables)
            # 更新梯度，这里参数的列表直接调用model.trainable_variables
            optimazer.apply_gradients(zip(grads, model.trainable_variables))
            # 每100个step打印一次信息
            if step % 100 ==0:
                print('epoch:{}\t step:{}\t loss_mse:{}\t loss_ce:{}\t'.format(epoch, step, float(loss_mse), float(loss_ce)))
        
        # 每个epoch计算一次准确率
        total_corrects = 0  # 统计变量
        total_number = 0
        # 针对validation dataset进行测试
        for x, y in validation_db:
            
            # 得到验证数据的预测结果
            probs = model(x)
    
            # 取最大值的索引为预测值
            preds = tf.cast(tf.argmax(probs, axis=1), dtype=tf.int32)
            
            # 累加正确的个数，和总数
            corrects = tf.equal(y, preds)
            corrects = tf.reduce_sum(tf.cast(corrects, dtype=tf.int32))
            total_corrects += corrects
            total_number += x.shape[0]
        
        # 计算测试数据集的准确率
        acc = total_corrects / total_number
        
        print('accuracy={};'.format(acc))
        
    # 两种保存model的方法都可以，low level：tf.saved_model.save(model, model_dir+'/'+datetime.now().strftime('%Y%m%d%H%M%S'))
    # 存储model的路径必须是数字类型的字符串
    model.save(model_dir+'/'+datetime.now().strftime('%Y%m%d%H%M%S'))

'''
用于在train()中，对datasets进行类型转换和进行归一化处理
paras：
    x：features
    y：labels
return：x，y
'''
def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

'''
获取训练数据集，从train数据集路径，通过pandas解析数据，然后分割成feature和label
最后组合成tensorflow dataset数据类型并返回
paras：
    train_data_path：train channel在sagemaker中的路径，examples是路径下的数据文件名，这里是硬coding需要改进
return：training的dataset
'''
def get_train_db(train_data_path):
    
    df=pd.read_csv(train_data_path+'/examples', sep=',',header=None)
    
    x_train, y_train = df.values[:, 1:], df.values[:, 0]
    
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    return train_db
   
def get_validation_db(validation_data_path):
    
    df=pd.read_csv(validation_data_path+'/examples', sep=',',header=None)
    
    x_validation, y_validation = df.values[:, 1:], df.values[:, 0]
    
    validation_db = tf.data.Dataset.from_tensor_slices((x_validation, y_validation))

    return validation_db

'''
解析sagemaker传递给这个训练脚本的参数，然后返回这些参数，供训练算法使用
'''
def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()

if __name__ == '__main__':
    
    # hello()
    
    # 获取参数
    args, unknown = parse_args()
    
    # 获取超参数
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    # 获取原始数据路径
    train_data_path = args.train
    validation_data_path = args.validation
    
    # 获取存储模型的路径
    model_dir = args.model_dir
    
    # 获取可以直接用于训练的datasets
    train_db = get_train_db(train_data_path)
    validation_db = get_validation_db(validation_data_path)
    
    # 获取模型
    model = get_model()
    
    # 开始训练
    train(model, train_db, validation_db, epochs, batch_size, learning_rate, model_dir)