wordEncode = ["good","thankyou","trouble","write"]
# try:
#     from tensorflow_docs.vis import embed
# except:
#     __import__("os").system("pip install -q git+https://github.com/tensorflow/docs")
#     from tensorflow_docs.vis import embed
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os
import uvicorn
import sys
from glob import glob
import cv2
import numpy as np
from fastapi import requests, FastAPI, File, UploadFile, BackgroundTasks
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse


html = ""
with open("./html/index.html",'r') as file:
    html = file.read()
results = dict()
height,width,chanel = 255,255,3
SEQUENCE_SIZE = 120
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def load_video(path, max_frames=0):
    cap = cv2.VideoCapture(path)
    frames = []
    dummy = np.zeros((height, width, chanel),dtype=np.uint8)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
              frame = cv2.resize(frame,(height,width))
              # frame = ndimage.rotate(frame, deg)
              frames.append(frame)
            if len(frames) == SEQUENCE_SIZE:
              break
        if SEQUENCE_SIZE-len(frames) != 0:
          for _ in range(SEQUENCE_SIZE-len(frames)):
            frames.append(dummy)

    finally:
        cap.release()
    return np.array(frames)[0::4]

# data = load_video("write.mp4")
# cv2.imshow("data",data[0])
# cv2.waitKey(0)

def build_feature_extractor():
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(height, width, chanel),
    )
    preprocess_input = keras.applications.densenet.preprocess_input
    inputs = keras.Input((height, width, chanel))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")
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
dataProcessor = build_feature_extractor()
model = keras.models.load_model('tryForBestv2.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding,"TransformerEncoder":TransformerEncoder})


def videoIn(path,id):
    vid = load_video(path)
    features = dataProcessor.predict(vid)
    pred = model.predict(features[None,...])
    results[id] = (wordEncode[np.argmax(pred)])


@app.get("/p/")
async def sendHtml():
    return HTMLResponse(html)
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    temp = await file.read()
    id = str(uuid4())
    path = f"./video/{id}.mp4"
    with open(path, 'a') as file:
        pass
    with open(path, 'wb') as file:
        file.write(temp)
    background_tasks.add_task(videoIn,path,id)
    return {"id": id}
@app.get("/getPrediction/")
async def createPred(id:str):
    try:
      out = {"out":results[id],"code":1}
    except:
        out = {"out":None,"code":0}
    return out
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", reload=True, port=8000)