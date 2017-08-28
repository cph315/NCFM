import inception_v4
import os
from keras.layers import Flatten, Dense, AveragePooling2D,GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
#from set_gpu import get_session
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
                    gpu_options=gpu_options, intra_op_parallelism_threads=num_threads
                    ))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session(0.5))

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 25
batch_size = 32

train_data_dir = '/home/julyedu_53998/study_SF/Kaggle_NCFM/data/train_split'
val_data_dir = '/home/julyedu_53998/study_SF/Kaggle_NCFM/data/val_split'

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

print('Loading InceptionV4 Weights ...')
InceptionV4_notop = inception_v4.create_model(include_top=False, weights='imagenet')
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
#output = InceptionV4_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = InceptionV4_notop.output  # Shape: (8, 8, 2048)
for i,layer in enumerate(InceptionV4_notop.layers):
    print(i,layer.name,layer.output_shape)
#output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = GlobalAveragePooling2D(name='Globalavg_pool')(output)
#output = Flatten(name='flatten')(output)
output = Dense(1024, activation='relu',name='fc_1')(output)
output = Dense(8, activation='softmax', name='predictions')(output)
print('Tom1+++++++++++++++++++++++++')
InceptionV4_model = Model(InceptionV4_notop.input, output)
#InceptionV3_model.summary()

optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV4_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
print('Tom2 model compiled++++++++++++++++++++++++')
# autosave best Model
best_model_file = "./inception_v4_weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visualization',
        # save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = '/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM/data/visulization',
        #save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

InceptionV4_model.fit_generator(
        train_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        nb_val_samples = nbr_validation_samples,
        callbacks = [best_model])
