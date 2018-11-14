
#Skin cancer classification

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(1322)
import itertools
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
import keras
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


skin_dir = os.path.join('..','input')

#combine into pictures into one directory
path_dir = {os.path.splitext(os.path.basename(x))[0]: x
           for x in glob(os.path.join(skin_dir,'*','*.jpg'))}

type_dict = {
      'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

# Create some new columns (path to image, human-readable name) and review them
meta_df['path'] = meta_df['image_id'].map(path_dir.get)
meta_df['cell_type'] = meta_df['dx'].map(type_dict.get)
meta_df['cell_type_idx'] = pd.Categorical(meta_df['cell_type']).codes
meta_df.sample(3)

meta_df.describe(exclude=[np.number])

#distibution of different cell types
fig,ax1 = plt.subplots(1,1,figsize=(10,5))
meta_df['cell_type'].value_counts().plot(kind='bar',ax=ax1)

#we need to balance this data
meta_df = meta_df.drop(meta_df[meta_df.cell_type_idx == 4].iloc[:5000].index)
fig,ax1 = plt.subplots(1,1,figsize=(10,5))
meta_df['cell_type'].value_counts().plot(kind='bar',ax=ax1)

#distribution of localizaiton of the images
meta_df['localization'].value_counts().plot(kind='bar')

#distrubtion of the sexes
meta_df['sex'].value_counts().plot(kind='bar')

meta_df['image'] = meta_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
#lets visaulize some of the images

n = 5
fig,m_axs = plt.subplots(7,n,figsize=(4*n,3*7))
for n_axs, (type_name,type_rows) in zip(m_axs,meta_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax,(_,c_row) in zip(n_axs,type_rows.sample(n,random_state=9).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png',dpi=300)
    
	
	#create the train and test sets of our data
y = meta_df.cell_type_idx
x_train_o,x_test_o,y_train_o,y_test_o = train_test_split(meta_df,y,test_size=0.25)

x_train = np.asarray(x_train_o['image']).tolist()
x_test = np.asarray(x_test_o['image']).tolist()

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std

#one hot
y_train = to_categorical(y_train_o,num_classes=7)
y_test = to_categorical(y_test_o,num_classes=7)
y_train_o.value_counts().plot(kind='bar')
y_test_o.value_counts().plot(kind='bar')


augs = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

augs.fit(x_train)
#annealer
annealer = ReduceLROnPlateau(monitor='val_loss',factor=0.2,
                            patience=3,min_lr=0.001)
input_shape = (75,100,3)
num_classes = 7

tmodel_base = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)

tmodel = Sequential()
tmodel.add(tmodel_base)
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.50))
tmodel.add(Flatten())
tmodel.add(Dense(512,activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.25))
tmodel.add(Dense(num_classes,activation='softmax',name='output_layer'))
tmodel.summary()

tmodel.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])
			  
			  batch_size = 128
epochs = 30

history = tmodel.fit(x_train,y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test,y_test),
                    verbose=1)
					
#evaluate the model
eval = tmodel.evaluate(x_test,y_test,verbose=0)
print('Test Loss:',eval[0])
print('Test Accuracy:',eval[1])

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

#confusion matrix
def plot_confusion_matrix(cm,classes,title='Confusion Matrix',
                         cmap=plt.cm.Blues,normaliza=False):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Correct Label')
    plt.xlabel('Predicted Label')
    
#predictions for validation
y_pred = tmodel.predict(x_test)
#predictions classes to one-hot
y_pred_classes = np.argmax(y_pred,axis=1)
#validation observations t one-hot
y_true = np.argmax(y_test,axis=1)
#build confusion matrix
confusion_mtx = confusion_matrix(y_true,y_pred_classes)
#plot confusion matrix
plot_confusion_matrix(confusion_mtx,classes = range(7))

