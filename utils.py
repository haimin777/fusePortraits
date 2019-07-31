import os, io, base64

import time
import simplekml

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import cv2

import exifread
import joblib



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# block for image manipulation

def clear_downloads(folder):
    for file in os.listdir(folder):
        os.remove(os.path.join(folder, file))
    print('uploads cleared')

def read_img(path):
    image = np.asarray(Image.open(path))
    image = cv2.resize(image, (512, 512))
    return image

def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None


def _convert_to_degress(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)

    return d + (m / 60.0) + (s / 3600.0)


def get_exif_location(exif_data):


    """
    Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)
    """
    lat = None
    lon = None

    gps_latitude = _get_if_exist(exif_data, 'GPS GPSLatitude')
    gps_latitude_ref = _get_if_exist(exif_data, 'GPS GPSLatitudeRef')
    gps_longitude = _get_if_exist(exif_data, 'GPS GPSLongitude')
    gps_longitude_ref = _get_if_exist(exif_data, 'GPS GPSLongitudeRef')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = _convert_to_degress(gps_latitude)
        if gps_latitude_ref.values[0] != 'N':
            lat = 0 - lat

        lon = _convert_to_degress(gps_longitude)
        if gps_longitude_ref.values[0] != 'E':
            lon = 0 - lon

    return lat, lon


def one_histogram(path, bins):
    arr = np.asarray(Image.open(path))

    return np.histogram(arr, bins)[0]


def get_results(path_to_image, classifier):
    image = one_histogram(path_to_image, 50)
    with open(path_to_image, 'rb') as f:
        coordinates = get_exif_location(exifread.process_file(f))

    predict = np.argmax(classifier.predict([image]))

    return coordinates, predict

def create_coords(app, remove_image=True):
    coords = []
    clf = joblib.load('drone_clf.pkl')
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        coords.append(get_results(img_path, clf))
        if remove_image:
            os.remove(img_path)

    return coords

def encode_images(align_fold='aligned_images', gen_fold='generated_images/', latent_fold='latent_representations/'):

    #time.sleep(15)
    clear_downloads(align_fold)
    os.system('python stylegan-encoder/align_images.py ./uploads aligned_images/')
    os.system('python stylegan-encoder/encode_images.py aligned_images/ generated_images/ latent_representations/')

    print('latents created!', '\n'*2)




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

