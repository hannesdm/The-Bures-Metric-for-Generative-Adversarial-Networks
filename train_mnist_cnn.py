import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np

from lib import data, models

# Create a CNN model trained on mnist - used for the evaluation of generative models on stacked mnist e.g. by mode counting

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
test_images = (test_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_images = np.reshape(train_images,(-1,28,28,1))
test_images = np.reshape(test_images,(-1,28,28,1))

cnn = models.mnist_cnn()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
checkpoint_path = 'data/cnn_model_checkpoint.hdf5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
mcp_save = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')
cnn.fit(train_images, train_labels, validation_data=(test_images,test_labels), callbacks=[earlyStopping, mcp_save],
        batch_size=32, epochs=100, verbose=2)
cnn.load_weights(checkpoint_path)
eval_out = cnn.evaluate(test_images, test_labels)
print(eval_out)

