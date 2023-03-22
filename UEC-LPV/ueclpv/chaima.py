from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np
from time import time
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#what does axis=0 mean? it is used for columns while axis=1 is used for rows
x_data= np.concatenate([x_train,x_test], axis=0)
y_lables= np.concatenate([y_train, y_test], axis=0)
print(x_data.shape)
print(y_lables.shape)
def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model
cae = CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10])
optimizer = 'adam'
cae.compile(optimizer=optimizer, loss='mse')
optimizer = 'adam'
cae.compile(optimizer=optimizer, loss='mse')
cae.load_weights('%s-pretrain-model-%d.h5' % ('mnist', 200))
print("weights have been loaded successfully")
# extract features
feature_model = Model(inputs=cae.input, outputs=cae.get_layer(name='embedding').output)
#feature_model = Model(cae.input, cae.get_layer(name='embedding').output)
features = feature_model.predict(x_data)
print('feature shape=', features.shape)
# use features for clustering
import fcmeans
from fcmeans import FCM
#number of cluster
n_clusters = len(np.unique(y_lables))
fcm = FCM(n_clusters=n_clusters)
features = np.reshape(features, newshape=(features.shape[0], -1))
print(features.shape)
fcm.fit(features)
# outputs
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)
from keras.losses import binary_crossentropy as bnc
import keras as k
import keras.backend as kb

class DFCM(k.Model):
    def __init__(self, encoder, decoder, centroids, **kwargs):
        super(DFCM, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.centroids = centroids

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            with tf.GradientTape() as tape:
                features = self.encoder(data)
                reconstruction = self.decoder(features)
                # computing reconstruction loss
                reconstruction_loss = tf.reduce_mean(
                    bnc(data, reconstruction)
                )
                reconstruction_loss *= 28 * 28
                # computing clustering loss
                # computing memberships
                # members= membership(features, self.centroids)
                members = 1 / kb.sum(kb.square(kb.expand_dims(features, axis=1) - self.centroids), axis=2)
                members **= 1 / (2 - 1)
                members = kb.transpose(kb.transpose(members) / kb.sum(members, axis=1))
                # computing new centroids
                # self.centroids= tf.Variable(update_centers(members, features))
                sq_mem = members ** 2
                centroid = kb.dot(kb.transpose(sq_mem), features)
                som = kb.sum(sq_mem, axis=0)
                centroid /= som[:, None]
                self.centroids = centroid#tf.Variable(centroid)
                # cls= cluster_loss(features,members, self.centroids)
                max_pos = kb.argmax(members, axis=1)
                #x= tf.ones(shape=(members.shape[1],1))
                x =0
                for i in range(0, members.shape[1]):
                     x+= kb.sum(kb.square(features[i,:] - self.centroids[max_pos[i],:]))
                    #x.append(kb.sum(kb.square(features[i, :] - self.centroids[max_pos[i], :])))
                cls = x/ members.shape[1] # statistics.mean(x)
                print("mean has been computed successfully")
                cls *= -0.5
                total_loss = reconstruction_loss + cls
        grads = tape.gradient(total_loss, self.trainable_weights)
        print("grads have been computed successfully")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        print("weights have been updated successfully")
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "clustering loss": cls,
        }
Inp_Encoder = Model(inputs=cae.input, outputs=cae.get_layer(name='embedding').output)
#building decoder model
from keras.layers import Input
decoded_inp=  Input(shape=(10,))
h=cae.layers[-5](decoded_inp)
h=cae.layers[-4](h)
h=cae.layers[-3](h)
h=cae.layers[-2](h)
h=cae.layers[-1](h)
Out_Decoder= Model(decoded_inp, h)
#Build DFCM model
dfcm = DFCM(encoder= Inp_Encoder, decoder= Out_Decoder,centroids=fcm_centers.astype(np.float32))
dfcm.compile(optimizer='adam')
dfcm.fit(x=x_data, y=x_data, shuffle= True,batch_size= 700, epochs=10)