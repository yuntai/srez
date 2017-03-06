"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

This app does the following:
    - Handles uploads and looks for an image file send as "file" parameter
    - Stores the image at ./images dir
    - Invokes ffwd_to_img function from evaluate.py with this image
    - Returns the output file generated at /output

Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: checkpoint
    - It is loaded from /input

Based on https://raw.githubusercontent.com/floydhub/fast-style-transfer/master/app.py
"""
import os
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from evaluate import ffwd_to_img

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
app = Flask(__name__)

def _srez_output(sess, input_path, feature, gene_output, output_path):
    size = [label.shape[1], label.shape[2]]

    clipped = tf.maximum(tf.minimum(gene_output, 1.0), 0.0)

    image = tf.concat(axis=2, values=[clipped, label])
    image = image[0:max_samples,:,:,:]
    image = tf.concat(axis=0, values=[image[i,:,:,:] for i in range(max_samples)])
    image = sess.run(image)

    scipy.misc.toimage(image, cmin=0., cmax=1.).save(output_path)
    print("    Saved %s" % (output_path,))

def __():
  feed_dict = {td.gene_minput: test_feature}
  gene_output = td.sess.run(td.gene_moutput, feed_dict=feed_dict)
  _srez_output()
  _summarize_progress(td, test_feature, test_label, gene_output, batch, 'out')

@app.route('/<path:path>', methods=["POST"])
def style_transfer(path):
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
        output_filepath = os.path.join('/output/', filename)
        file.save(input_filepath)

        # Get checkpoint filename from la_muse
        checkpoint = request.form.get("checkpoint") or "la_muse.ckpt"
        ffwd_to_img(input_filepath, output_filepath, '/input/' + checkpoint, '/gpu:0')
        return send_file(output_filepath, mimetype='image/jpg')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')
