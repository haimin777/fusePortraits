#!/usr/bin/env python

import os, io, base64
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
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

#import exifread
#import joblib

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DOWNLOAD_FOLDER'] = 'downloads/'

# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'gif', 'JPG'])

# For a given file, return whether it's an allowed type or not

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# block for image manipulation


@app.route('/')
def index():
    return render_template('index.html')


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():

    clear_downloads('/home/bizon/CBIS-DDSM/other/fuse_face_flask/uploads')

    # Get the name of the uploaded files
    # Get the name of the uploaded file
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
        # testcv.image_make(filename)

        # return redirect(url_for('uploaded_file', filename=filename))
        return render_template('second_download.html')


@app.route('/upload2', methods=['POST'])
def upload2():

    # Get the name of the uploaded files
    # Get the name of the uploaded file
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
                # testcv.image_make(filename)

                # return redirect(url_for('uploaded_file', filename=filename))
                #return render_template('second_download.html')

                print('redirected to process file')
                return redirect(url_for('preprocess_images'))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload


#@app.route('/uploads/<filename>')
@app.route('/preprocess_images', methods=['POST'])

def uploaded_file(filename):
    render_template('encode_images.html')
    #coords = create_coords(app)
    encode_images()

    #create_kml(coords, app)
    # try to plot
    '''
    chart_plot = Markup(
        '<img style="padding:1px; border:1px solid #021a40; width: 100%; height: 100%" src="data:image/png;base64,{}">'.format(
            create_chartplot_url(coords)))
    '''
    #return render_template('uploaded.html', title='Home', chart1_plot=None)
    return render_template('images_processed.html')



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
        #return send_file('downloads/polygon_0.kml')
        chart_plot = Markup(
        '<img style="padding:1px; border:1px solid #021a40; width: 100%; height: 100%" src="data:image/png;base64,{}">'.format(
            create_portret_url(gen_img)))
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