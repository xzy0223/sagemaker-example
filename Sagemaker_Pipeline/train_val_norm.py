import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql.functions import *
from awsglue.context import GlueContext
from awsglue.job import Job

# 获取Glue Job传进来的参数
args = getResolvedOptions(sys.argv, ['JOB_NAME','SOURCE_PATH', 'OUTPUT_PATH', 'TRAIN_PREFIX', 'VAL_PREFIX'])

# 获取Spark Context运行环境并生成Glue运行环境
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

# 开始Job
job = Job(glueContext)
job.init(args['JOB_NAME'], args)


'''
对featurn数据进行normalization的函数
参数：
    data_source_url：原始数据的S3路径
    data_output_url：存储处理过的数据的S3路径
'''
def norm_transform(data_source_url, data_output_url):
    
    # 读取原始数据为dataframe
    source_data_frame = spark.read.load(data_source_url, format='csv',
                                        inferSchema=True, header=False)
    
    # MNIST数据集包含785列，第一列为label，剩下的为feature，选择dataframe的第一列数据生成新的label dataframe
    source_label_data_frame = source_data_frame.select(source_data_frame.columns[0])
    
    # 丢掉第一列，剩下的feature生成feature dataframe
    source_feature_data_frame = source_data_frame.drop(source_data_frame.columns[0])
    
    # 获得feature所有列的列表
    columns = source_feature_data_frame.columns
    # 遍历所有的列，对数据进行normalization
    for column in columns:
        source_feature_data_frame = source_feature_data_frame.withColumn(column, (source_feature_data_frame[column] / 255.))

    # 对feature和label数据分别生成自增id，两个dataframe的id是完全一样的
    source_label_data_frame = source_label_data_frame.withColumn("id", monotonically_increasing_id())
    source_feature_data_frame = source_feature_data_frame.withColumn("id", monotonically_increasing_id())
    
    # 通过outer join的方式将两组dataframe在列的方向进行合并，并删除不在需要的id
    target_train_data_frame = source_label_data_frame.join(source_feature_data_frame, "id", "outer").drop("id")

    # 如果想存储的数据只生成一个文件，那么可以repartition为1
    #target_train_data_frame = target_train_data_frame.repartition(1)
    
    # 存储数据到S3
    target_train_data_frame.write.save(
        data_output_url, 
        format='csv', 
        mode='overwrite')

# 组织好训练数据和验证数据
train_data_source_url = args['SOURCE_PATH'] + args['TRAIN_PREFIX'] + '*'
train_data_output_url = args['OUTPUT_PATH'] + args['TRAIN_PREFIX']

val_data_source_url = args['SOURCE_PATH'] + args['VAL_PREFIX'] + '*'
val_data_output_url = args['OUTPUT_PATH'] + args['VAL_PREFIX']

# 进行数据转换
norm_transform(train_data_source_url, train_data_output_url)
norm_transform(val_data_source_url, val_data_output_url)

# 提交Job
job.commit()
