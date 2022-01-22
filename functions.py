
from library import *

def building_model(inp,input_shape = (128,128,3)):
    filters = [64,128,256]
    x = inp
    for i in filters:
        x = Conv2D(filters=i, kernel_size=3, padding='same', activation='relu', input_shape=input_shape)(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    return x

def convert_to_precintage(predict):
    a = []
    total_size = predict.shape[-1]
    for i in predict:
        t = (a-np.min(i))/(np.max(i)-np.min(i))
        t = np.sum(t)/total_size
        a.append((1-t) * 100)

    return a

def mse(a):
    data = []
    for i in a:
        t = np.sum(i)
        data.append(t/512.0)
    return np.array(data)

def normlize(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))

