import argparse
import os
from datetime import datetime

import tensorflow as tf


# 配置TF的日志级别，0:无屏蔽，1:屏蔽info，2:屏蔽warning及以下，3:屏蔽error及以下
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 获取 TF dataset 的方法
def get_dataset(data_path, batch_size):
    """
    详细描述：
    params
    ----------
    data_path: str
        数据所在的路径
    batch_size: int
        生成的数据集每个batch的record数量
    """
    file = os.listdir(data_path)
    
    # api guide： https://tensorflow.google.cn/api_docs/python/tf/data/experimental/make_csv_dataset?hl=en
    # input csv文件需要有header，返回一个dataset，其中每个元素是一个（特征，标签）元组。特征字典将特征列名映射相应特征数据的Tensors，label是包含该批次标签数据的Tensor
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=data_path+'/'+file[0],
        batch_size=batch_size,
        label_name='label',
        na_value="0",
        num_epochs=1,
        ignore_errors=True)
    return dataset


def get_model():
    
    # 观影种类特征的字典
    genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
                   'Sci-Fi', 'Drama', 'Thriller',
                   'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

    GENRE_FEATURES = {
        'userGenre1': genre_vocab,
        'userGenre2': genre_vocab,
        'userGenre3': genre_vocab,
        'userGenre4': genre_vocab,
        'userGenre5': genre_vocab,
        'movieGenre1': genre_vocab,
        'movieGenre2': genre_vocab,
        'movieGenre3': genre_vocab
    }

    # all categorical features
    categorical_columns = []
    for feature, vocab in GENRE_FEATURES.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        emb_col = tf.feature_column.embedding_column(cat_col, 10)
        categorical_columns.append(emb_col)
    # movie id embedding feature
    movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
    movie_emb_col = tf.feature_column.embedding_column(movie_col, 10)
    categorical_columns.append(movie_emb_col)

    # user id embedding feature
    user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
    user_emb_col = tf.feature_column.embedding_column(user_col, 10)
    categorical_columns.append(user_emb_col)

    # all numerical features
    numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
                         tf.feature_column.numeric_column('movieRatingCount'),
                         tf.feature_column.numeric_column('movieAvgRating'),
                         tf.feature_column.numeric_column('movieRatingStddev'),
                         tf.feature_column.numeric_column('userRatingCount'),
                         tf.feature_column.numeric_column('userAvgRating'),
                         tf.feature_column.numeric_column('userRatingStddev')]

    # cross feature between current movie and user historical movie
    rated_movie = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=1001)
    crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([movie_col, rated_movie], 10000))

    # define input for keras model
    inputs = {
        'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
        'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
        'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
        'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
        'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
        'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
        'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

        'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
        'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
        'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),

        'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
        'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
        'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
        'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
        'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
        'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
        'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
        'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
    }

    # wide and deep model architecture
    # deep part for all input features
    deep = tf.keras.layers.DenseFeatures(numerical_columns + categorical_columns)(inputs)
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    deep = tf.keras.layers.Dense(128, activation='relu')(deep)
    # wide part for cross feature
    wide = tf.keras.layers.DenseFeatures(crossed_feature)(inputs)
    both = tf.keras.layers.concatenate([deep, wide])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(both)
    model = tf.keras.Model(inputs, output_layer)  
    
    return model

def train(model, train_data_path, validation_data_path, epochs, batch_size, learning_rate, model_dir):
    
    train_db = get_dataset(train_data_path, batch_size)
    test_db = get_dataset(validation_data_path, batch_size)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, name='Adam')
    
    # compile the model, set loss function, optimizer and evaluation metrics
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

    # train the model
    model.fit(train_db, epochs=epochs, verbose=2)

    # evaluate the model
    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_db, verbose=0)
    print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                       test_roc_auc, test_pr_auc))

    # print some predict results
    predictions = model.predict(test_db)
    print(predictions)
    print(list(test_db)[0][1][:12])
    for prediction, goodRating in zip(predictions[:12], list(test_db)[0][1][:12]):
        print("Predicted good rating: {:.2%}".format(prediction[0]),
              " | Actual rating label: ",
              ("Good Rating" if bool(goodRating) else "Bad Rating"))
        
    model.save(model_dir+'/'+datetime.now().strftime('%Y%m%d%H%M%S'))

'''
解析sagemaker传递给这个训练脚本的参数，然后返回这些参数，供训练算法使用
'''
def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()

    
if __name__ == '__main__':
    
    # 获取参数
    args, unknown = parse_args()
    
    # 获取超参数
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    # 获取训练数据路径
    train_data_path = args.train
    validation_data_path = args.validation
    
    # 获取存储模型的路径
    model_dir = args.model_dir
    
    # 获取模型
    model = get_model()
    
    # 开始训练
    train(model, train_data_path, validation_data_path, epochs, batch_size, learning_rate, model_dir)