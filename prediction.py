from library import *
from functions import *

PATH = pr.convertor(r"D:\Data\faces")
print(PATH)
  
d = pr.data(PATH)
# pr.vis(d)

all_data, pair_labels = pr.pair_data(d) 
_, test_x, _, test_y = train_test_split(all_data, pair_labels, test_size=0.2, random_state=52) 

model = tf.keras.models.load_model('face_similarity/SNN_w1.h5')
pred_1 = model.predict([test_x[:,0],test_x[:,1]],batch_size=32,verbose=1)

compare_image = np.array([np.hstack(i) for i in test_x])

print(pred_1)
print(pred_1.shape)
plt.figure(figsize=(15,18))
for index,i in enumerate(compare_image[:25]):
    plt.subplot(5,5,index+1)
    plt.imshow(i)
    plt.title(f"{pred_1[index][1]*100:.2f}%")

plt.show()

model = tf.keras.models.load_model('face_similarity/SNN_w1.h5')
model_1 = Model(inputs = model.layers[1].input, outputs = model.layers[-3].output)
pred2 = model_1.predict(test_x[:,0],batch_size=32,verbose=1)
pred3 = model_1.predict(test_x[:,1],batch_size=32,verbose=1)

print(pred2)
print(pred2)
t = mse(pred2)
print(t)
plt.figure(figsize=(15,18))
for index,i in enumerate(compare_image[:25]):
    plt.subplot(5,5,index+1)
    plt.imshow(i)
    s = ssim(pred2[index],pred3[index])
    print(type(s))
    plt.title(f"{s:.4f}")

plt.show()