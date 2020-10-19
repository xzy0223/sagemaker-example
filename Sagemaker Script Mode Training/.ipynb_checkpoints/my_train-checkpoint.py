import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, metrics, Sequential


# 数据处理函数，对x和y进行数据预处理，比如对x进行归一化或者对y进行onehot转换等
def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# load数据，返回train和test数据,数据类型是numpy array
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# 分别将train和test数据转换成tensorflow的dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# 对dataset进行处理
# 预处理传入process函数
# shuffle用于把数据打散，参数越大打的越散
# 设定每一个batch的大小
train_db = train_db.map(process).shuffle(10000).batch(128)
test_db = test_db.map(process).shuffle(10000).batch(128)

# 获得sample数据，查看数据的shape信息，用于下一步定义网络的一些参数
train_iter = iter(train_db)
train_sample = next(train_iter)
print('train dataset x shape {}, train dataset y shape {}'.format(train_sample[0].shape, train_sample[1].shape))

# 使用keras的Sequential类，创建一个容器，其中定义每一层的Layer
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),  # w:[784, 256], b:[256], 激活函数使用relu
    layers.Dense(128, activation=tf.nn.relu),  # w:[256, 128], b:[128], 激活函数使用relu
    layers.Dense(64, activation=tf.nn.relu),  # w:[128, 64], b:[64], 激活函数使用relu
    layers.Dense(32, activation=tf.nn.relu),  # w:[64, 32], b:[32], 激活函数使用relu
    layers.Dense(10)  # w:[64, 10], b:[10], 最后一层一般不使用激活函数，直接输出数值，也叫做logits，为了数据稳定性
])

# build网络，指定输入数据的格式，这个输入的shape一定要跟真实数据一致
model.build(input_shape=[None, 28*28])
# 打印出网络的汇总信息，每层类型和参数等
model.summary()
# optimazer用于更新参数，优化loss，设定learning rate
optimazer = optimizers.Adam(lr=1e-3)

# 使用main函数包裹训练算法
def main():
    # 设定epoch的次数
    for epoch in range(1):
        
        for step, (x, y) in enumerate(train_db):
            # 对input进行reshape，对应model build的input_shape
            x = tf.reshape(x, [-1,28*28])
            # tape包裹前向运算，用于记录varibale，好计算梯度
            with tf.GradientTape() as tape:
                # 直接将x输入model，实际上调用的实例的__call__方法
                logits = model(x) 
                # 对y进行onehot编码，因为y的shape是[b,],而logits的shape是[b, 10],需要将y转换成[b, 10]
                y_hot = tf.one_hot(y, depth=10)
                # 这里设置两个损失函数mse和交叉熵
                # 注意，如果是使用logits计算交叉熵的化，需要指定参数from_logits=True
                loss_mse = tf.reduce_mean(tf.losses.mean_squared_error(y_hot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_hot, logits, from_logits=True))
            # 计算对于交叉熵的参数的梯度   
            grads = tape.gradient(loss_ce, model.trainable_variables)
            # 更新梯度，这里参数的列表直接调用model.trainable_variables
            optimazer.apply_gradients(zip(grads, model.trainable_variables))
            # 每100个step打印一次信息
            if step % 100 ==0:
                print('epoch:{}\t step:{}\t loss_mse:{}\t loss_ce:{}\t'.format(epoch, step, float(loss_mse), float(loss_ce)))
                print("abcd\tefgh\txhy")
        
        # 每个epoch计算一次准确率
        total_corrects = 0  # 统计变量
        total_number = 0
        # 针对test dataset进行测试
        for x, y in test_db:
            
            x = tf.reshape(x, [-1,28*28])
            
            logits = model(x)
            # 使用softmax转换logits为probs
            probs = tf.nn.softmax(logits)
            # 取最大值的索引为预测值
            preds = tf.cast(tf.argmax(probs, axis=1), dtype=tf.int32)
            
            # 累加正确的个数，和总数
            corrects = tf.equal(y, preds)
            corrects = tf.reduce_sum(tf.cast(corrects, dtype=tf.int32))
            total_corrects += corrects
            total_number += x.shape[0]
        
        # 计算测试数据集的准确率
        acc = total_corrects / total_number
        
        print('accuracy={}'.format(acc))

if __name__ == '__main__':
    main()