# Resources:
# https://huggingface.co/transformers/model_doc/roberta.html#tfrobertamodel
# https://www.tensorflow.org/tutorials/distribute/keras
# https://www.kaggle.com/xhlulu/jigsaw-tpu-xlm-roberta

###################################################################
# IMPORTS Pt.1
###################################################################
print("running imports, pt.1...")

import os
print(os.getcwd())
print(os.listdir(os.getcwd()))

###################################################################
# INSTALLS
###################################################################
print("running installs...")

os.system('pip install transformers')

###################################################################
# IMPORTS Pt.2
###################################################################
print("running imports...")

import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# TF Imports
import tensorflow as tf
print("tf version: ", tf.__version__)

# Keras Imports
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger

# Sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Garbage Collector
import gc
import sys

import transformers
# from transformers import TFAutoModel, AutoTokenizer
# from transformers import RobertaTokenizer, TFRobertaModel
from transformers import DistilBertTokenizer, TFDistilBertModel
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors

# Boto is the Amazon Web Services (AWS) SDK for Python, which allows Python developers to write software that makes
# use of Amazon services like S3 and EC2. Boto provides an easy to use, object-oriented API as well as low-level direct
# service access.
import boto3
#import s3fs
# args
import argparse

###################################################################
# HELPER FUNCTIONS
###################################################################
print("defining helper functions...")


def parse_args():
    parser = argparse.ArgumentParser()
    # data directories
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    return parser.parse_known_args()


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids'])


def build_model(transformer, max_len=512):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def get_train_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "jigsaw_mjy_train_val_openaug_523200.csv"))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return train, test, sub

if __name__ == "__main__":
    
    ###################################################################
    # SETTINGS
    ###################################################################
    # Detect hardware, return appropriate distribution strategy
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None
    
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.MirroredStrategy()
        
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    
    ###################################################################
    # CONSTANTS
    ###################################################################
    print("setting constants...")
    
    # Configuration
    MAX_LEN = 192
    #MODEL = 'jplu/tf-xlm-roberta-large'
    #MODEL = 'roberta-base'
    MODEL = 'distilbert-base-multilingual-cased'
    myOutput = '/opt/ml/model/'
    expCounter = 1
    AUTO = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    EPOCHS = 2
    
    ###################################################################
    # TOKENIZER
    ###################################################################
    print("loading tozenizer...")
    
    # First load the real tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(MODEL)
    #tokenizer = RobertaTokenizer.from_pretrained(MODEL)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL)
    
    ###################################################################
    # DATA
    ###################################################################
    print("loading training data...")

    args, _ = parse_args()
    
    train, test, sub = get_train_data(args.data)
    x_train = regular_encode(train.comment_text.values, tokenizer, maxlen=MAX_LEN)
    x_test = regular_encode(test.content.values, tokenizer, maxlen=MAX_LEN)
    y_train = train.toxic.values
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=1331)
    
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    valid_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )
    
    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices(x_test)
        .batch(BATCH_SIZE)
    )

    ###################################################################
    # LOAD MODEL
    ###################################################################
    print("loading model ...")
    
    with strategy.scope():
        #transformer_layer = TFAutoModel.from_pretrained(MODEL)
        transformer_layer = TFDistilBertModel.from_pretrained(MODEL)
        model = build_model(transformer_layer, max_len=MAX_LEN)
    model.summary()
    
    ###################################################################
    # TRAINING
    ###################################################################
    print("run training ...")
    
    n_steps = x_train.shape[0] // BATCH_SIZE
    
    train_history = model.fit(
        train_dataset,
        steps_per_epoch=n_steps,
        validation_data=valid_dataset,
        epochs=EPOCHS
    )
    
    ###################################################################
    # SCORE
    ###################################################################
    print("score test data ...")
    sub['toxic'] = model.predict(test_dataset, verbose=1)
    sub.to_csv('submission.csv', index=False)
    
    ###################################################################
    # COMPLETE
    ###################################################################
    
    print("Program complete!")
    