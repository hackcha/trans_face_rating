from vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.optimizers import SGD,Adam
from keras.layers import Conv2D, Dropout, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, concatenate
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras import applications
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
import numpy as np
from keras.utils.np_utils import to_categorical
import data_utils
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.regularizers import l1
import matplotlib.pyplot as plot


def get_encoder():
    encoder = applications.VGG19(include_top=False, input_shape=(224, 224, 3))
    #Flatten(input_shape=encoder.output_shape[1:])
    h1 = GlobalMaxPooling2D()(encoder.output)
    h2 = GlobalAveragePooling2D()(encoder.output)
    h = concatenate( [h1, h2] )
    model = Model(encoder.input, h)
    return model

def def_model( nb_class ):
    input = Input(shape=(1024,))
    reg = l1(1e-10)
    # h = BatchNormalization()(input)
    h = Dense( 64, activation='relu',kernel_regularizer=reg, bias_regularizer=reg)(input)#
    h = BatchNormalization()(h)
    # h = Dropout(0.5)(h)
    y = Dense( nb_class, activation='softmax' )( h )#,kernel_regularizer=reg, bias_regularizer=reg
    model = Model( input, y )
    return model

def raw_model( nb_class ):
    input = Input(shape=(224, 224, 3))
    h1 = Conv2D(filters=50,kernel_size=11,strides=(4,4), activation='relu')(input)
    h1_p = MaxPooling2D(pool_size=(2,2),strides=(2,2))(h1)
    h2 = Conv2D(filters=100, kernel_size=5, activation='relu')(h1_p)
    h2_p = MaxPooling2D(pool_size=(2,2),strides=(2,2))(h2)
    h2_g = GlobalMaxPooling2D()(h2_p)
    reg = l1(1e-10)
    z1 = Dense(64, activation='relu',kernel_regularizer=reg, bias_regularizer=reg)(h2_g)
    z1 = BatchNormalization()(z1)
    # z1 = Dropout(0.5)(z1)
    y = Dense(nb_class, activation='softmax')(z1)
    model = Model(input, y)
    return model


def def_model_reg( nb_class ):
    input = Input(shape=(1024,))
    reg = l1(1e-8)
    h = Dense(64, activation='relu', kernel_regularizer=reg, bias_regularizer=reg,kernel_initializer='normal', bias_initializer='normal')(input)
    h = BatchNormalization()(h)
    y = Dense(1,kernel_regularizer=reg, bias_regularizer=reg, kernel_initializer='normal', bias_initializer='normal')(h)
    model = Model(input, y)
    return model


def train(model, xs, ys, model_path, opt=None):
    if opt is None:
        opt='adam'
    model.compile(opt,'categorical_crossentropy', metrics=['accuracy'])
    model.summary( )
    earlystop = EarlyStopping(patience=20)
    save_best = ModelCheckpoint(model_path)
    model.fit( xs, ys, batch_size=32, validation_split=0.1, epochs=50, callbacks=[earlystop, save_best])
    model.load_weights(model_path)
    # model.save_weights(model_path)


def test( encoder, model, img_path, nb_class=None ):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    xe = encoder.predict(x)
    print(img_path)
    print('Input image shape:', x.shape)
    probs = model.predict(xe)
    if nb_class is None:
        print('Predicted:', probs)
    else:
        y = probs.argmax(axis=-1)
        print('Predicted:', y )

def test_raw( model, img_path, nb_class=None ):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(img_path)
    print('Input image shape:', x.shape)
    probs = model.predict(x)
    if nb_class is None:
        print('Predicted:', probs)
    else:
        y = probs.argmax(axis=-1)
        print('Predicted:', y )

def cls_model_main():
    xs, xse, ys, ys2, ys3, ys5 = data_utils.load_cache( )
    encoder = get_encoder( )
    nb_class = 5
    ys = ys5
    ys = to_categorical(ys, nb_class)
    model = def_model(nb_class)
    model_path = 'data/model_{}.h5'.format(nb_class)
    opt = SGD(lr=0.5,decay=0.9999, clipnorm=5.0)#0.85,0.72
    # opt = SGD(lr=0.3, decay=0.9999, clipnorm=5.0)  # 0.86,0.66
    # opt = SGD(lr=0.5, clipnorm=5.0)  # 0.97,0.70
    # opt = SGD(lr=0.3, clipnorm=5.0)  # 0.96,0.70
    # opt = Adam(lr=0.0005, clipnorm=5.0)#1.0,0.64
    # opt = Adam(lr=0.0001, clipnorm=5.0)  # 1.0,0.60
    #for dropout
    # opt = Adam(lr=0.0001, clipnorm=5.0)  # 0.67,0.70
    # opt = Adam(lr=0.0005, clipnorm=5.0)  # 0.67,0.70
    # opt = Adam(lr=0.001, clipnorm=5.0)  # 0.74,0.70
    # opt = SGD(lr=0.9, decay=0.99999, clipnorm=5.0)  # 0.67,0.70
    train(model, xse, ys, model_path, opt)
    img_path = 'data/Data_Collection/SCUT-FBP-1.jpg'
    test(encoder, model, img_path, nb_class)
    img_path = 'data/Data_Collection/SCUT-FBP-45.jpg'
    test(encoder, model, img_path, nb_class)
    img_path = 'data/Data_Collection/SCUT-FBP-110.jpg'
    test(encoder, model, img_path, nb_class)
    img_paths = ['data/lyf_ce.jpg',
                 'data/lyf_z.jpg',
                 'data/rh.jpg',
                 'data/tw.jpg']
    for img_path in img_paths:
        test(encoder, model, img_path, nb_class)

def reg_model_main():
    nb_class=None
    xs, xse, ys, ys2, ys3, ys5 = data_utils.load_cache( )
    # ys = (ys-1)/4
    encoder = get_encoder( )
    model = def_model_reg(nb_class)
    model_path = 'data/model_reg.h5'
    # this optimization is suitable for using relu activation, with BN, val_loss ~ 0.25
    opt = SGD(0.5,decay=0.999, clipnorm=5.0)
    # this optimization setting is suitable for using sigmoid activation, without BN, val_loss~0.25
    # opt = SGD(0.01, clipnorm=5.0)
    model.compile(opt, 'mean_squared_error')
    model.summary()
    earlystop = EarlyStopping(patience=5)
    model.fit(xse, ys, batch_size=32, validation_split=0.1, epochs=20, callbacks=[earlystop])
    model.save_weights(model_path)
    img_path = 'data/Data_Collection/SCUT-FBP-1.jpg'
    test(encoder, model, img_path, nb_class)
    img_path = 'data/Data_Collection/SCUT-FBP-45.jpg'
    test(encoder, model, img_path, nb_class)
    img_path = 'data/Data_Collection/SCUT-FBP-110.jpg'
    test(encoder, model, img_path, nb_class)

def cls_raw_model_main():
    xs, xse, ys, ys2, ys3, ys5 = data_utils.load_cache()
    nb_class = 5
    ys = ys5
    ys = to_categorical(ys, nb_class)
    model = raw_model(nb_class)
    model_path = 'data/raw_model_{}.h5'.format(nb_class)
    #easy to over fitting
    # opt = Adam(lr=0.001,clipnorm=5.0)#1.0,0.72
    # opt = Adam(lr=0.0001, clipnorm=5.0)#1.0,0.64
    # opt = Adam(lr=0.0005, clipnorm=5.0)#1.0,0.72
    #opt = SGD(lr=0.5,decay=0.9999, momentum=0.95,clipnorm=5.0)#0.71，0.70
    #opt = SGD(lr=0.5, decay=0.9999, momentum=0.99, clipnorm=5.0)#0.66,0.70
    opt = SGD(lr=0.5, decay=0.9999, clipnorm=5.0)#0.77,0.68
    # opt = SGD(lr=0.9, decay=0.9999, clipnorm=5.0)#0.75,0.64
    # opt = SGD(lr=0.3, decay=0.9999, clipnorm=5.0)  # 0.85,0.60
    # opt = SGD(lr=0.3, decay=0.99999, clipnorm=5.0)  # 0.91,0.66
    #dropout
    # opt = Adam(lr=0.0005, clipnorm=5.0)  # 0.67,0.70
    # opt = SGD(lr=0.3, decay=0.99999, clipnorm=5.0)#0.67,0.70
    train(model, xs, ys, model_path,opt)
    img_path = 'data/Data_Collection/SCUT-FBP-1.jpg'
    test_raw( model, img_path, nb_class)
    img_path = 'data/Data_Collection/SCUT-FBP-45.jpg'
    test_raw( model, img_path, nb_class)
    img_path = 'data/Data_Collection/SCUT-FBP-110.jpg'
    test_raw( model, img_path, nb_class)

def load_and_test():
    nb_class = 5
    model_path = 'data/model_{}.h5'.format(nb_class)
    encoder = get_encoder()
    model = def_model(nb_class)
    model.load_weights(model_path)
    img_paths = ['data/lyf_ce.jpg',
                 'data/lyf_z.jpg',
                 'data/rh.jpg',
                 'data/tw.jpg']

    for img_path in img_paths:
        test(encoder, model, img_path, nb_class)
    model_path = 'data/raw_model_{}.h5'.format(nb_class)
    rmodel = raw_model(nb_class)
    rmodel.load_weights(model_path)
    for img_path in img_paths:
        test_raw(rmodel, img_path, nb_class)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    # cls_model_main( )
    # reg_model_main( )
    # cls_raw_model_main()
    load_and_test( )


