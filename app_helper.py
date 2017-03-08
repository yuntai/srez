import os
import tensorflow as tf
from scipy import misc
import uuid

import srez_model
from srez_main import setup_tensorflow
import srez_input

FLAGS = tf.app.flags.FLAGS
FLAGS.batch_size = 1
FLAGS.train_dir = "./train"
FLAGS.checkpoint_dir = "/mnt/tmp/checkpoint"

def get_input_feature(input_fn, out_dir="."):
  with tf.Session() as sess:
    img = misc.imread(input_fn)
    print("img.shape=", img.shape)
    channels = img.shape[-1]
    sz = min(list(img.shape)[:2])
    image = tf.Variable(img)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.image.resize_image_with_crop_or_pad(image, sz, sz)
    image = tf.reshape(image, [1, sz, sz, channels])
    sess.run(tf.global_variables_initializer())
    img, = sess.run([image])
  tf.reset_default_graph()
  outfn = os.path.join(out_dir, str(uuid.uuid4())+".jpg")
  misc.toimage(img.reshape(list(img.shape)[1:]), cmin=0., cmax=1.).save(outfn)
  return img, sz, outfn

def _downsample(input_filepath, output_filepath):
  channels = 3; K = 4

  img = misc.imread(input_filepath)
  h, w, _ = img.shape

  image = tf.Variable(img)
  image = tf.reshape(image, [1, h, w, channels])
  image = tf.cast(image, tf.float32)/255.0
  image = tf.image.resize_area(image, [h//K, w//K])
  #image = tf.image.resize_nearest_neighbor(image, [h, w])
  image = tf.reshape(image, [h//K, w//K, channels])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    image_out = sess.run(image)

  misc.toimage(image_out, cmin=0., cmax=1.).save(output_filepath)

def srez_output(input_fn, output_fn, checkpoint_dir):
  input_image, sz, hackfn = get_input_feature(input_fn)
  # dummy files to satisfy input pipeline
  filenames = ['000001.jpg']

  sess, summary_writer = setup_tensorflow()
  try:
    features, labels = srez_input.setup_inputs(sess, filenames, sz*4)
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    feed_dict = {gene_minput: input_image}
    gene_output = sess.run(gene_moutput, feed_dict=feed_dict)
    gene_output = gene_output.reshape(list(gene_output.shape)[1:])
    misc.toimage(gene_output, cmin=0., cmax=1.).save(output_fn)
  finally:
    sess.close()

if __name__ == "__main__":
  TEST_INPUT='/mnt/tmp/dn-051357.jpg'
  TEST_OUTPUT='/mnt/tmp/output0.jpg'
  TEST_CHECKPOINT_DIR='/mnt/tmp/checkpoint'
  srez_output(TEST_INPUT, TEST_OUTPUT, TEST_CHECKPOINT_DIR)
