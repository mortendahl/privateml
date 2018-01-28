from keras.utils import to_categorical
from keras.datasets import mnist


def preprocess_data(dataset):
    
    (x_train, y_train), (x_test, y_test) = dataset
    
    # NOTE: this is the shape used by Tensorflow; other backends may differ
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    x_train  = x_train.astype('float32')
    x_test   = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    y_train = to_categorical(y_train, 5)
    y_test  = to_categorical(y_test, 5)
    
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_public = x_train[y_train < 5]
    y_train_public = y_train[y_train < 5]
    x_test_public  = x_test[y_test < 5]
    y_test_public  = y_test[y_test < 5]
    public_dataset = (x_train_public, y_train_public), (x_test_public, y_test_public)

    x_train_private = x_train[y_train >= 5]
    y_train_private = y_train[y_train >= 5] - 5
    x_test_private  = x_test[y_test >= 5]
    y_test_private  = y_test[y_test >= 5] - 5
    private_dataset = (x_train_private, y_train_private), (x_test_private, y_test_private)
    
    return preprocess_data(public_dataset), preprocess_data(private_dataset)