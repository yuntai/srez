"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes ffwd_to_img function from evaluate.py with this image
    - Returns the output file generated at /output

Based on https://raw.githubusercontent.com/floydhub/fast-style-transfer/master/app.py
"""
import os
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

import app_helper

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)

CHECKPOINT_FILE='/input/checkpoint/checkpoint_new.txt'

os.makedirs('./images', exist_ok=True)
os.makedirs('./output', exist_ok=True)

@app.route('/<path:path>', methods=["POST"])
def srez(path):
    """
    Take the input image and style transfer it
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        return BadRequest("File not present in request")
    file = request.files['file']
    if file.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(file.filename):
        return BadRequest("Invalid file type")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        input_filepath = os.path.join('./images/', filename)
        output_filepath = os.path.join('./output/', filename)

        file.save(input_filepath)
        print("output_filepath=",output_filepath)

        if 'downsample' == request.form.get('run'):
          app_helper.downsample(input_filepath, output_filepath)
        else:
          # run = 'srez' or None
          app_helper.srez_output(input_filepath, output_filepath, CHECKPOINT_FILE)

        # Get checkpoint filename from la_muse
        return send_file(output_filepath, mimetype='image/jpg')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')


