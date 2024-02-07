import numpy as np

from keras.models import Sequential  # 采用贯序模型
from keras.layers import Input, Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import time

# 指定GPU训练
import os

# 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=[28, 28, 1]))  # 第一卷积层
    model.add(Conv2D(64, (5, 5), activation='relu'))  # 第二卷积层
    model.add(MaxPool2D(pool_size=(2, 2)))  # 池化层
    model.add(Flatten())  # 平铺层
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def compile_model(model):
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 优化函数，设定学习率（lr）等参数
    # 多GPU训练
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
    return model


def train_model(model, x_train, y_train, batch_size=128, epochs=10):
    # 构造一个tensorboard类的对象
    # tbCallBack = TensorBoard(log_dir="./model", histogram_freq=1, write_graph=True, write_images=True,update_freq="epoch")

    tbCallBack = TensorBoard(log_dir="./model", histogram_freq=1, write_grads=True)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                        validation_split=0.2, callbacks=[tbCallBack])
    return history, model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # mnist的数据我自己已经下载好了的
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    print(np.shape(x_train), np.shape(y_train), np.shape(x_test), np.shape(y_test))

    model = create_model()
    model = compile_model(model)
    print("start training")
    ts = time.time()
    history, model = train_model(model, x_train, y_train)
    print("start training", time.time() - ts)