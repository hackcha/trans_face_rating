from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def rate2class( rate, nb_class ):
    if nb_class == 2:
        if rate<3:
            return 0
        else:
            return 1
    if nb_class == 3:
        if rate < 2.3:
            return 0
        elif rate < 3.6:
            return 1
        else:
            return 2
    #rate from 1-5
    if rate > 4.2:
        return 4
    return int(rate-1.0)


def convert_data( ):
    xs = []
    ys = []
    ys2 = []
    ys3 = []
    ys5 = []
    with open( 'data/Attractiveness_label.csv', 'r') as fin:
        fin.readline()
        for line in fin:
            terms = line.strip().split(',')
            index = int(terms[0].strip())
            rate = float(terms[1].strip() )
            img_path = 'data/Data_Collection/SCUT-FBP-{}.jpg'.format(index)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            # x = np.expand_dims(x, axis=0)
            ys.append( rate )
            y2 = rate2class( rate, 2 )
            ys2.append( y2 )
            y3 = rate2class( rate, 3)
            ys3.append( y3 )
            y5 = rate2class(rate, 5 )
            ys5.append( y5 )
            xs.append( x )
    xs = np.asarray( xs )
    xs = preprocess_input(xs)
    import model
    encoder = model.get_encoder()
    xse = encoder.predict( xs )
    print(xse.shape)
    ys = np.array( ys )
    ys2 = np.array( ys2 )
    ys3 = np.array( ys3 )
    ys5 = np.array( ys5 )
    np.savez('data/cache.npz', xs = xs, xse=xse, ys = ys, ys2 = ys2, ys3 = ys3, ys5 = ys5)

def load_cache(fname = 'data/cache.npz'):
    r = np.load( fname )
    xs, xse, ys, ys2,ys3,ys5 = r['xs'], r['xse'], r['ys'], r['ys2'], r['ys3'], r['ys5']
    return xs, xse, ys, ys2,ys3,ys5

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    set_session(sess)
    convert_data( )
    print('convert data done.')