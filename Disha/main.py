from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import sys

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.2
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu),layers.BatchNormalization(),layers.Dense(256, activation=tf.nn.relu),layers.BatchNormalization(),layers.Dense(1024, activation=tf.nn.relu),layers.BatchNormalization() ,layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
data = None
with open(sys.argv[1],'rb') as file:
    data = pickle.load(file)

f,_ = data.shape
try:
    pred = np.concatenate([data,np.zeros(shape=(120-f,1024),dtype=np.float32)])
except:
    pred = data[:120]
tempDict = ["good","write","thankyou","trouble"]
model1 = keras.models.load_model('newVersion.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding,"TransformerEncoder":TransformerEncoder})
print(model1.summary())
print(pred[None,...].shape)
pred = model1.predict(pred[None,...])
res = np.argmax(pred)
print(res)
print(tempDict[res])