"""
Select and Train Model
    Vectorizes training and validation texts
    training a Some model
    Evaluate model using Cross-Validation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import tensorflow as tf

from text_classification.build_model import mlp_model
from text_classification.load_data import load_cnews_dataset
from text_classification.vectorize_data import ngram_vectorize
from text_classification.explore_data import get_num_classes


FLAGS = None


def train_ngram_model(data,
                      model_dir,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        model_dir: string, path of model will be saved.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_texts, train_labels), (val_texts, val_labels),  (test_texts, test_labels) = data

    num_classes = get_num_classes(train_labels)

    # Vectorize texts.
    x_train, x_val, x_test = ngram_vectorize(
        train_texts, train_labels, val_texts, test_texts, model_dir)

    # convert sparse matrix to dense
    x_train = x_train.toarray()
    x_val = x_val.toarray()
    x_test = x_test.toarray()

    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=units,
                                  dropout_rate=dropout_rate,
                                  input_shape=x_train.shape[1:],
                                  num_classes=num_classes)

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    print(x_train.shape)
    print(train_labels.shape)
    print(x_val.shape)
    print(val_labels.shape)
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save(os.path.join(model_dir, 'cnews_mlp_model.h5'))

    # evaluate model
    loss, accuracy = model.evaluate(x_test, test_labels, batch_size=32)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='input data directory')
    parser.add_argument('--model_dir', type=str, default='./saved_model',
                        help='path to the location of saved model')

    FLAGS, unparsed = parser.parse_known_args()

    if not tf.io.gfile.exists(FLAGS.model_dir):
        tf.io.gfile.mkdir(FLAGS.model_dir)

    # Using the cnews dataset to training n-gram model
    data = load_cnews_dataset(FLAGS.data_dir)
    train_ngram_model(data, model_dir=FLAGS.model_dir)
