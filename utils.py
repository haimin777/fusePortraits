import os, io, base64
import shutil as sh
import time
#import simplekml

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/bizon/CBIS-DDSM/other/fuse_face_flask/stylegan-encoder')
import glob
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# block for image manipulation

def clear_downloads(folder):
    for file in os.listdir(folder):
        try:
            os.remove(os.path.join(folder, file))
        except:
            sh.rmtree(os.path.join(folder, file))
    print('uploads cleared')

def read_img(path):
    image = np.asarray(Image.open(path))
    image = cv2.resize(image, (512, 512))
    return image

def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None




def one_histogram(path, bins):
    arr = np.asarray(Image.open(path))

    return np.histogram(arr, bins)[0]


def get_results(path_to_image, classifier):
    image = one_histogram(path_to_image, 50)
    with open(path_to_image, 'rb') as f:
        coordinates = get_exif_location(exifread.process_file(f))

    predict = np.argmax(classifier.predict([image]))

    return coordinates, predict


def encode_images(align_fold='aligned_images', gen_fold='generated_images/', latent_fold='latent_representations/'):

    #time.sleep(15)
    clear_downloads('/home/bizon/CBIS-DDSM/other/fuse_face_flask/stylegan-encoder/aligned_images')
    clear_downloads('/home/bizon/CBIS-DDSM/other/fuse_face_flask/stylegan-encoder/generated_images')
    clear_downloads('/home/bizon/CBIS-DDSM/other/fuse_face_flask/stylegan-encoder/latent_representations')
    
    os.chdir('stylegan-encoder')
    os.system('python align_images.py /home/bizon/CBIS-DDSM/other/fuse_face_flask/uploads aligned_images/')
    os.system('python encode_images.py aligned_images/ generated_images/ latent_representations/ --iterations 200')

    print('latents created!', '\n'*2)
    
    
    
def generate_image(latent_vector):
    os.chdir('/home/bizon/CBIS-DDSM/other/fuse_face_flask/stylegan-encoder')
    print('dir changed')

    #################################
        # init generator #
    URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    #################################

    os.chdir('/home/bizon/CBIS-DDSM/other/fuse_face_flask')
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))





def create_kml(coordinates, app):
    kml = simplekml.Kml()

    for class_n in [0, 1, 2]:
        coord_for_poly = [coord[0] for coord in coordinates if coord[1] == class_n]
        x = [coord[0] for coord in coord_for_poly]
        y = [coord[1] for coord in coord_for_poly]

        center_point = [np.sum(coord_for_poly[0]) / len(coord_for_poly[0]),
                        np.sum(coord_for_poly[1]) / len(coord_for_poly[1])]
        angles = np.arctan2(x - center_point[0], y - center_point[1])
        sort_tups = sorted([(i, j, k) for i, j, k in zip(x, y, angles)], key=lambda t: t[2])
        polygon_coords = [(coord[0], coord[1]) for coord in sort_tups]
        kml.newpolygon(name="mypoly" + '_' + str(class_n), outerboundaryis=polygon_coords)

    kml.save(app.config['DOWNLOAD_FOLDER'] + '/polygon_0.kml')

def create_chartplot_url(coords):
    x = [coord[0][0] for coord in coords]
    y = [coord[0][1] for coord in coords]
    z = [coord[1] for coord in coords]

    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, marker='o', alpha=0.5)
    plt.suptitle('Drone photos map')
    dif = (max(x) - min(x)) * 0.1

    ax.set_autoscaley_on(False)
    ax.set_ylim([min(y) - dif, max(y) + dif])
    ax.set_autoscalex_on(False)
    ax.set_xlim([min(x) - dif, max(x) + dif])

    labels = ['' for _ in ax.get_xticklabels()]
    labels[1] = round(x[0], 4)
    labels[-2] = round(x[-1], 4)
    ax.set_xticklabels(labels)

    labels_y = ['' for item in ax.get_yticklabels()]
    labels_y[1] = round(y[0], 4)

    labels_y[-2] = round(y[-1], 4)
    ax.set_yticklabels(labels_y)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()  # return plot_url


def create_portret_url(gen_img):
   
    fig, ax = plt.subplots()
    ax.imshow(gen_img)
    plt.suptitle('generated photo')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()  # return plot_url

