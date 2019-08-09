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
    return img




def create_portret_url(gen_img):
   
    fig, ax = plt.subplots()
    ax.imshow(gen_img)
    plt.suptitle('generated photo')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()  # return plot_url

def uploaded_photos_url(raw_folder='uploads', npy=False):
    
    if npy:
        plt.subplot(121).imshow(np.load(os.path.join(raw_folder, os.listdir(raw_folder)[0])))
        plt.subplot(122).imshow(np.load(os.path.join(raw_folder, os.listdir(raw_folder)[1])))
        plt.suptitle('generated latents vectors (18x512)')
    else:
        
        plt.subplot(121).imshow(np.array(PIL.Image.open(os.path.join(raw_folder, os.listdir(raw_folder)[0]))))
        plt.subplot(122).imshow(np.array(PIL.Image.open(os.path.join(raw_folder, os.listdir(raw_folder)[1]))))
        plt.suptitle('uploaded photos')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()  # return plot_url

