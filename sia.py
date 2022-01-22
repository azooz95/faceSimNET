from library import *
from functions import *

tf.executing_eagerly()

tnp.experimental_enable_numpy_behavior()

PATH = pr.convertor(r"D:\Data\faces")
print(PATH)
  
d = pr.data(PATH,number_of_folder=1)
# pr.vis(d)
print(d.shape)

all_data, pair_labels = pr.pair_data(d) 

train_x, test_x, train_y, test_y = train_test_split(all_data, pair_labels, test_size=0.2, random_state=52) 

print(f"train_x:{train_x.shape}, train_y:{train_y.shape}, test_x: {test_x.shape}, test_y:{test_y.shape}")

SHAPE_IMG = (128,128,3)

left_input = Input(shape=SHAPE_IMG)
right_input = Input(shape=SHAPE_IMG)

left_model = building_model(left_input)
right_model = building_model(right_input)

com_layer = Lambda(lambda Tensors: tf.keras.backend.abs(Tensors[0] - Tensors[1]))

com_layer = com_layer([left_model,right_model])
prediction = Dense(1,activation='sigmoid')(com_layer)

snn_model = Model(inputs=[left_input, right_input], outputs=prediction)

print(snn_model.summary())
snn_model.compile(loss="binary_crossentropy" , optimizer='adam', metrics=['accuracy'])

epoch = 50
batch_size = 16
callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0)

snn_model.fit([train_x[:,0], train_x[:,1]],train_y,
            validation_data=([test_x[:,0], test_x[:,1]], test_y),
            batch_size=batch_size,
            epochs=epoch,
            callbacks=callback,
            verbose=1)
snn_model.save('face_similarity/SNN_w2.h5')



# plot_model(snn_model, to_file='face_similarity/SNN.png', show_shapes=True)


# plot data 
# split data to training test 
# building the model 
# loss function 
