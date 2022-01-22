# import numpy as np 
# import cv2 as cv
# import matplotlib.pyplot as plt
# from scipy.special import softmax

# # import tensorflow_probability as tfp
# import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity
# # from tensorflow.python.keras.backend import sigmoid
# def difrence(x,y):
#     return np.abs(x-y)
# def sigomid(a):
#     return np.where(a<0.5,0,1)
# path_1 = "D:\\Data\\faces\\00000\\00000.png"
# path_2 = "D:\\Data\\faces\\00000\\00011.png"

# img1 = cv.imread(path_1)
# img2 = cv.imread(path_2)
# a = img1.flatten()/255.0
# b = img2.flatten()/255.0

# a = tf.convert_to_tensor(a,dtype=np.float32)
# b = tf.convert_to_tensor(b,dtype=np.float32)


# print(a.shape)
# result = difrence(a,b)
# print(result)
# result = 100 - (sum(result) * 100) 
# print(result)

# t = np.array([1,9,2,0,0,0,1,10])
# y = (t-min(t))/(max(t)-min(t))
# print(sum(y)/len(y))
# j = np.array([9,10,10,10,10,10,10,10])
# y = (j-min(j))/(max(j)-min(j))
# print(sum(y)/len(y))

# def normlize(a):
#     return (a-min(a))/(max(a)-min(a))
# # print(a.shape)
# # size = a.shape[0]
# # print(np.sum(cosine_similarity(a,b))/size)
# # # print(R3)
# # print(R3[0,1])
# # plt.imshow(img)
# # plt.show()

# # x = tf.random.normal(shape=(100))
# # y = tf.random.normal(shape=(100))

# # # corr[i, j] is the sample correlation between x[:, i, j] and y[:, i, j].
# # corr = tfp.stats.correlation(x, y)

# # # corr_matrix[i, m, n] is the sample correlation of x[:, i, m] and y[:, i, n]
# # corr_matrix = tfp.stats.correlation(x, y, sample_axis=0, event_axis=-1)

# # print(corr)
# print(type(a))
# from skimage.metrics import structural_similarity as ssim
# import tensorflow.experimental.numpy as tnp
# x = lambda Tensors: tf.expand_dims(tf.convert_to_tensor(ssim(
#                                                             tnp.asarray(Tensors[0]),
#                                                             tnp.asarray(Tensors[1])),np.float32),axis=0)




# # import tensorflow.experimental.numpy as tnp
# # x = lambda Tensors: tf.expand_dims(tf.convert_to_tensor(ssim(
# #                                                             a,
# #                                                             b),np.float32),axis=0)
# print(x([a,b]))

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# path = "D:\\Data\\archive\\VGG16v7.csv"
# data = pd.read_csv(path, index_col=False)
# plt.plot(data.index, data['accuracy'])
# plt.plot(data.index, data['val_accuracy'])
# plt.legend(['Training accuracy','Testing accuracy'])
# plt.grid()
# plt.title("Accuracy Curve")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.show()


# import numpy as np 
# import cv2 as cv
# import matplotlib.pyplot as plt 

# path = "D:\\Downloads\\cas.jpg"
# value = 30
# img = cv.imread(path)

# limit = 255 - value
# img_new = np.where(img>limit,img+value,img)
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.subplot(1,2,2)
# plt.imshow(img_new)
# plt.show()

# import pandas as pd 
# import numpy as np 

# a = pd.DataFrame({1:[1,2,3,4], 2:[5,6,9,8]})
# t = np.linspace(start=11,stop=14,num=3)
# t = range(0,10)
# print(list(t))
# e = np.array([t,t])
# print(e.shape)
# a.loc[0:1,list(t)]=e
# print(a)
# import math
# h = 0
# m = 0
# def make_readable(seconds):
#     global h,m
#     s = seconds
#     if seconds>=3600:
#         h,seconds = math.floor(seconds/3600), seconds%3600
#     elif seconds>=60:
#         m,seconds = math.floor(seconds/60), seconds%60
#     else:
#         return str(h).zfill(2)+":"+str(m).zfill(2)+":"+str(seconds).zfill(2)
#     return make_readable(seconds)

# time = make_readable(137591)
# print(time)

# def Rot13(messsage):
#     s = []
#     for i in messsage: 
#         if i.isalpha():
#             if i.isupper():
#                 s.append(chr(ord('A')+(((ord(i)-ord('A'))+13)%26)))
#             else:
#                 s.append(chr(ord('a')+(((ord(i)-ord('a'))+13)%26)))
#         else:
#             s.append(i)
#     print(s)
#     return "".join(s)
# s = Rot13("tesT d e e fr test")
# print(s)
