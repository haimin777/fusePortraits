#!/usr/bin/env python

import os, io, base64, sys

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Markup, send_file
from werkzeug import secure_filename
#import simplekml


from utils import *


import flask
import numpy as np
import pandas as pd
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DOWNLOAD_FOLDER'] = 'downloads/'

app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'gif', 'JPG'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def download_one_file():
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        print(filename)


@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    
    if len(os.listdir('uploads'))>=2:

        clear_downloads('/home/bizon/CBIS-DDSM/other/fuse_face_flask/uploads')
        print('clear all', '\n')
        download_one_file()

        # return redirect(url_for('uploaded_file', filename=filename))
        return render_template('second_download.html')
    else:
        download_one_file()
        downloaded_photos = uploaded_photos_url()
        raw_imgs_plot = Markup(
        '<img style="padding:1px; border:1px solid #021a40; width: 100%; height: 100%" src="data:image/png;base64,{}">'.format(
            downloaded_photos))
        return render_template('redy_to_proc.html', chart_2_raw_imgs_plot=raw_imgs_plot)
      
            

        



@app.route('/preprocessimages')
def preprocess_images():
    sys.stdout.flush()
    encode_images()
    latents_imgs_plot = Markup(
        '<img style="padding:1px; border:1px solid #021a40; width: 100%; height: 100%" src="data:image/png;base64,{}">'.format(
            uploaded_photos_url(raw_folder='latent_representations', npy=True)))
    sys.stdout.flush()
    return render_template('images_processed.html', chart1_plot=latents_imgs_plot)



@app.route('/downloads/')
def return_files_tut():
    try:
        latents_path = glob.glob('/home/bizon/CBIS-DDSM/other/fuse_face_flask/stylegan-encoder/latent_representations/*.npy')


        latents = np.load(latents_path[0])
        latents_d = np.load(latents_path[1])
        latents+=latents_d
        latents = latents/2
        gen_img = generate_image(latents)
        print('start sending!', '\n'*2)
        chart_plot = Markup(
        '<img style="padding:1px; border:1px solid #021a40; width: 100%; height: 100%" src="data:image/png;base64,{}">'.format(
            create_portret_url(gen_img)))
        sys.stdout.flush()
        return render_template('uploaded.html', title='Home', chart1_plot=chart_plot)

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(
        #host="127.0.0.1",
        host="0.0.0.0",        
        port=int("5000"),
        debug=True
)
