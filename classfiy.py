import numpy as np
np.random.seed(1)
from keras.datasets import mnist
from keras.utils import np_utils    
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

# download the mnist to the path 
# X.shape = (60,000 28*28) Y.shape = (10,000, )
(X_train,y_train),(X_test,y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0],-1)/255
X_test = X_test.reshape(X_test.shape[0],-1)/255
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# Another way to build your netural net
model = Sequential([
    Dense(32,input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax') ,
])

# Another way to define your optimizer
rmsprop = RMSprop(lr = 0.001, rho = 0.9, epsilon= 1e-08, decay=0.0)

# We add netrics to get more rsults you want to see
model.compile(
    optimizer = rmsprop, 
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'],
)

print("Traing--------------------------------------------")
# Another way to train the model
model.fit(X_train,y_train,epochs = 20,batch_size=32)

print("\n Testing--------------------------------------------------")
# Evaluate the model with the metrics we defined earlier
loss,accuracy = model.evaluate(X_test,y_test)

print('test loss:',loss)
print("test accuracy:",accuracy)
