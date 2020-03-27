#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[3]:


# Intialising The CNN:
classifier = Sequential()


# In[4]:


#Step-1 Convolution:
classifier.add(Convolution2D(filters = 32, kernel_size = (3,3), activation = "relu", kernel_initializer = "he_uniform", input_shape = (32,32,3)))


# In[5]:


#step - 2 MaxPooling:
classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[6]:


#step - 3 Flattening:
classifier.add(Flatten())


# In[7]:


# Step - 4 Full Connections:
classifier.add(Dense(units = 128, activation = "relu", kernel_initializer = "he_uniform"))
classifier.add(Dense(units = 1, activation = "sigmoid"))


# In[8]:


#compile:
classifier.compile(optimizer= "Adam", loss= "binary_crossentropy", metrics = ["accuracy"])


# In[9]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[9]:


# !pip install pillow


# In[10]:


#Part-2 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'C:/Users/KARNDEEP SINGH/Desktop/Machine Learning A-Z/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/P14-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/training_set',
        target_size=(32,32),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'C:/Users/KARNDEEP SINGH/Desktop/Machine Learning A-Z/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/P14-Convolutional-Neural-Networks/Convolutional_Neural_Networks/dataset/test_set',
        target_size=(32, 32),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(train_generator,
                        steps_per_epoch=8000,
                        epochs=5,
                        validation_data=validation_generator,
                        validation_steps=2000)


# In[11]:


classifier.save("model.h5")


# In[15]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:\\Users\\KARNDEEP SINGH\\Downloads\\smartest-dog-breeds-1553287693.jpg', target_size = (32, 32))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
# train.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)


# In[ ]:




