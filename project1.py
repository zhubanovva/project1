#!/usr/bin/env python
# coding: utf-8

# # Home Work Model: Run 1

# ## Spark

# In[1]:


#import os
# Set spark environments
#os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3.6'
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-class-path /data-public/common_resources/ojdbc6-11.1.0.7.0.jar --jars /data-public/common_resources/ojdbc6-11.1.0.7.0.jar pyspark-shell '

# In[2]:


# creating spark session
#import findspark
#findspark.init('/opt/spark3')
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .config('spark.dynamicAllocation.enabled', 'true') \
    .config('spark.master', 'yarn') \
    .config("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true") \
    .config("spark.sql.debug.maxToStringFields", 1000) \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true") \
    .config('spark.sql.sources.partitionOverwriteMode','dynamic') \
    .config('hive.exec.dynamic.partition.mode', 'nonstrict') \
    .config('hive.exec.dynamic.partition', 'true') \
    .appName('home_work') \
    .enableHiveSupport() \
    .getOrCreate()

# ## Libraries

# In[4]:


# python libraries
import numpy as np
import pandas as pd

# time and datetime
from time import time
from time import sleep
from datetime import date, timedelta, datetime

# pyspark functions and data types
from pyspark.sql.window import Window
from pyspark.sql import functions as f
from pyspark.sql import types as t

# pyspark settings 
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1") # faster joins
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true") # to prevent problems with killing while writing a table
spark.conf.set('spark.sql.sources.partitionOverwriteMode','dynamic') # additional config needed to enable dynamic writing to partitions
spark.conf.set('hive.exec.dynamic.partition.mode', 'nonstrict') # additional config needed to enable dynamic writing to partitions
spark.conf.set('hive.exec.dynamic.partition', 'true') # additional config needed to enable dynamic writing to partitions

# logging and traceback
import logging
logging.basicConfig(format="%(asctime)s %(message)s")
logging.getLogger().setLevel(logging.INFO)
import traceback



# ## Dictionaries

# In[5]:


PATHS = {
    'table1': 'hive1.table1',
    'stg_1': 'dbzhubanova.stg_1'  
}
DLC_COLS_INFO = [
    ('COLUMN1', 'long'),
    ('COLUMN2', 'string'),
    ('COLUMN3', 'long'),
    ('COLUMN4', 'long'),
    ('COLUMN5', 'string'),
    ('COLUMN6', 'string'),
    ('COLUMN7', 'double'),
    ('COLUMN8', 'double'),
]
STG_1_COLS_INFO = [
    ('COLUMN9', 'long'),
    ('COLUMN10', 'string'),
    ('COLUMN11', 'string'),
    ('COLUMN12', 'string'),
    ('COLUMN13', 'string'),
    ('COLUMN14', 'string'),
    ('COLUMN3', 'long'),
    ('COLUMN4', 'long'),
    ('DATE', 'string'),
]


# ## Helper Functions

# In[6]:


def check_table(df):
    print('Schema')
    print(df.printSchema())
    print('#'*50)
    print('Count: ', df.count())
    print('#'*50)
    print('Show')
    print(df.show())


def cache_and_trigger(df):
    
    """function to save spark dataframe to cache and trigger it instantly"""
    
    df = df.cache()
    df.count()
    return df



def table_exists(table_path):
    database, table = table_path.split('.')
    tables = [table.name for table in spark.catalog.listTables(database)]
    return table in tables



def read_hive_table(table_name, repartition=False, num_of_partitions=10):
    """
    this function read a table from Hive
    """
    # logging
    logging.info(f'reading {table_name} from Hive')
    if repartition:
        return spark.read.table(table_name).repartition(num_of_partitions)
    else:
        return spark.read.table(table_name)
    # logging
    logging.info('successful read')
    
def write_hive_table(df, table_name, mode='overwrite'):
    """
    this function write a given dataframe into a specified Hive table
    """
    # logging
    logging.info(f'writing into {table_name} in the mode {mode}')
    df.write.mode(mode).saveAsTable(table_name)
    # logging
    logging.info('successful write')
    
def create_or_insert(df, col_partition, write_path):
    """
    this function partitions and saves data using either creating a new table or inserting into existing table
    """
    if not table_exists(write_path):
        df.write.partitionBy(col_partition).saveAsTable(write_path)
        logging.warning(f'{write_path} does not exists. Created it.')
    else:
        df.write.mode('overwrite').insertInto(write_path, overwrite=True)
    
def delete_partition(table_path, col_partition, val_partition):
    """
    this function deletes a partition with a corresponding value
    """
    logging.info(f'started deletion of partition. Table: {table_path}. Partition: {col_partition}={val_partition}')
    if not table_exists(table_path):
        logging.warning(f'there is no table on {table_path}. No partition to delete')
        return
    partition_list = spark.sql(f"SHOW PARTITIONS {table_path}").toPandas().partition.tolist()
    partition_keys = [item.split('=')[0] for item in partition_list]
    partition_values = [item.split('=')[1] for item in partition_list]
    partition_key = partition_keys[0]
    if partition_key != col_partition:
        logging.warning('col_partition name is not correct. partition_key={partition_key}. col_partition={col_partition}')
        raise ValueError('col partition name is not correct')
    if not val_partition in partition_values:
        logging.warning(f'there is not partition {col_partition}={val_partition}')
        return
    spark.sql(f"""ALTER TABLE {table_path} DROP IF EXISTS PARTITION ({col_partition}='{val_partition}')""")
    logging.info(f'For the table {table_path} the partition {col_partition}={val_partition} is deleted')
    logging.info(f'completed deletion of partition. Table: {table_path}. Partition: {col_partition}={val_partition}')
    
def get_all_days(date, history_days):
    """
    this function return a list of dates up to the given 'date' and number of days equal to 'history_days'
    Args:
    - date: string in yyyy-mm-dd formar
    - history_days: int 
    Returns:
    - list of dates in yyyy-mm-dd string format
    """
    date_range = pd.date_range(end=date, periods=history_days+1, closed='left')
    result = [item.strftime('%Y-%m-%d') for item in date_range]
    return result

def get_table1(date_str, paths):
    """
    this function get table1 data from Hive for the given date
    Args:
    - date_str: date in yyyy-mm-dd format, e.g. 2020-02-02
    - paths: dict with paths
    """
    path = paths.get('table1')
    df = read_hive_table(path)
    df = df.withColumn('START_DATE_STR', f.date_format(f.col('start_date'), 'yyyy-MM-dd'))
    df = df.withColumn('END_DATE_STR', f.date_format(f.col('end_date'), 'yyyy-MM-dd'))
    cond_relevant = ((f.col('START_DATE_STR') <= date_str) & (f.col('END_DATE_STR') >= date_str))
    df = df.filter(cond_relevant)
    # check count
    if df.count() == 0:
        raise ValueError('No rows in table1')
    
    # prepare columns
    df = df.withColumn('COLUMN1', f.col('COLUMN1').cast('long'))
    df = df.withColumn('COLUMN4', f.col('column4').cast('long'))
    df = df.withColumn('COLUMN3', f.col('COLUMN3').cast('long'))
    df = df.withColumn('COLUMN7', f.col('column8').cast('double'))
    df = df.withColumn('COLUMN8', f.col('column8').cast('double'))
    df = df.withColumn('COLUMN21', f.col('column21'))

    # filters
    cond_bs_valid =(
    f.col('COLUMN1').isNotNull() & \
    (f.col('COLUMN3') > 0) & \
    (f.col('COLUMN4') > 0) & \
    (f.col('COLUMN7').isNotNull()) & \
    (f.col('COLUMN8').isNotNull()))
    df = df.filter(cond_bs_valid)

    # cast types
    for col_name, col_type in DLC_COLS_INFO:
        df = df.withColumn(col_name, f.col(col_name).cast(col_type))

    # selects
    selects = [item[0] for item in DLC_COLS_INFO]
    df = df.select(*selects)
    df = df.dropDuplicates(subset=['COLUMN1'])
    return df
    

def get_stg_1(date, paths, cols_info, dlc):
    """
    this function prepares data for stg_1
    """
    # logging
    logging.info(f'started computing daily stg_1 for {date}')
    old = get_table1(date=date, paths=paths, cols_info=cols_info)
    new = get_table1(date=date, paths=paths, cols_info=cols_info)
    df = old.union(new)
    # logging
    logging.info(f'completed computing daily stg_1 for {date}')
    return df_clean

def insert_stg_1(dates, cols_info, paths):
    """
    this function inserts data to stg_1
    """
    # logging
    logging.info('started inserting data to stg_1')
    path_stg_1 = paths.get('stg_1')
    # start a loop
    for cur_date in dates:
        dlc = get_table1(cur_date, paths)
        stg_1 = get_stg_1(cur_date, paths, cols_info, dlc)
        # inserting
        create_or_insert(stg_1, 'DATE', path_stg_1)
        # logging
        logging.info(f'stg_1 data for {cur_date} is inserted to {path_stg_1} table')
    # logging
    logging.info('completed inserting data to stg_1')
    


def write_stg_1(date_key, history_days, paths, cols_info):
    """
    this function writes data for stg_1
    """
    # logging
    logging.info(f'started writing overall stg_1 for date_key {date_key}')
    dates = get_all_days(date_key, history_days)
    # find dates to delete and add
    path_stg_1 = paths.get('stg_1')
    if table_exists(path_stg_1):
        partition_list = spark.sql(f"SHOW PARTITIONS {path_stg_1}").toPandas().partition.tolist()
        partition_dates = [item.split('=')[1] for item in partition_list]
    else:
        partition_dates = []
    del_partitions = [item for item in partition_dates if item not in dates]
    add_partitions = [item for item in dates if item not in partition_dates]
    # delete not needed partitions
    delete_stg_1(del_partitions, paths)
    # add needed partitions
    insert_stg_1(add_partitions, cols_info, paths)
    # logging
    logging.info(f'completed writing overall stg_1 for date_key {date_key}')
    
    
def main(date_key, history_days):
    """
    main function for work and home model
    """
    # logging 
    logging.info(f'started main function for work_home model for date_key {date_key}')
    logging.info('#'*50)
    write_stg_1(date_key, history_days, PATHS, STG_1_COLS_INFO)
    # logging
    logging.info('#'*50)
    logging.info(f'completed main function for work_home model for date_key {date_key}')


# ## Action

# In[12]:
if __name__ == "__main__":
    DATE_KEY=date.today().strftime('%Y-%m-%d')
    HISTORY_DAYS=60
    main(DATE_KEY, HISTORY_DAYS)

spark.stop()
