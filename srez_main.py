import srez_demo
import srez_input
import srez_model
import srez_train

import os.path
import random
import numpy as np
import numpy.random

import tensorflow as tf

import zipfile
import uuid
import sys

FLAGS = tf.app.flags.FLAGS

# Configuration (alphabetically)
tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Number of samples per batch.")

tf.app.flags.DEFINE_string('checkpoint_dir', '/output/checkpoint',
                           "Output folder where checkpoints are dumped.")

tf.app.flags.DEFINE_integer('checkpoint_period', 10000,
                            "Number of batches in between checkpoints")

tf.app.flags.DEFINE_string('dataset', '',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_string('dataset_zip', '/input/data/img_align_celeba.zip',
                           "Path to the dataset directory.")

tf.app.flags.DEFINE_float('epsilon', 1e-8,
                          "Fuzz term to avoid numerical instability")

tf.app.flags.DEFINE_string('run', 'demo',
                            "Which operation to run. [demo|train]")

tf.app.flags.DEFINE_float('gene_l1_factor', .90,
                          "Multiplier for generator L1 loss term")

tf.app.flags.DEFINE_float('learning_beta1', 0.5,
                          "Beta1 parameter used for AdamOptimizer")

tf.app.flags.DEFINE_float('learning_rate_start', 0.00020,
                          "Starting learning rate used for AdamOptimizer")

tf.app.flags.DEFINE_integer('learning_rate_half_life', 5000,
                            "Number of batches until learning rate is halved")

tf.app.flags.DEFINE_bool('log_device_placement', False,
                         "Log the device where variables are placed.")

tf.app.flags.DEFINE_integer('sample_size', 64,
                            "Image sample size in pixels. Range [64,128]")

tf.app.flags.DEFINE_integer('summary_period', 200,
                            "Number of batches between summary data dumps")

tf.app.flags.DEFINE_integer('random_seed', 0,
                            "Seed used to initialize rng.")

tf.app.flags.DEFINE_integer('test_vectors', 16,
                            """Number of features to use for testing""")

tf.app.flags.DEFINE_string('train_dir', '/output/train',
                           "Output folder where training logs are dumped.")

tf.app.flags.DEFINE_string('demo_output_dir', '/output',
                           'Output folder where demo video is dumped.')

tf.app.flags.DEFINE_integer('train_time', 20,
                            "Time in minutes to train the model")

tf.app.flags.DEFINE_integer('train_batch_iterations', -1,
                            "Time in minutes to train the model")

tf.app.flags.DEFINE_string('device', '/gpu:0',
                            "tensorflow device to use /cpu:0 or /gpu:0")

def prepare_dirs(delete_train_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

    # Cleanup train dir
    if delete_train_dir:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)

    if FLAGS.dataset:
      # Return names of training files
      if not tf.gfile.Exists(FLAGS.dataset) or \
         not tf.gfile.IsDirectory(FLAGS.dataset):
          raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))
    else:
      if not tf.gfile.Exists(FLAGS.dataset_zip):
          raise FileNotFoundError("Could not find zip file `%s'" %
                                  (FLAGS.dataset_zip,))

      dataset_dir = "/" + str(uuid.uuid4())
      tf.gfile.MakeDirs(dataset_dir)
      local_zip = os.path.join(dataset_dir, os.path.basename(FLAGS.dataset_zip))
      print("Copying %s to %s"%(FLAGS.dataset_zip, local_zip,))
      sys.stdout.flush()
      tf.gfile.Copy(FLAGS.dataset_zip, local_zip)
      with zipfile.ZipFile(local_zip) as zf:
        print("Extracting %s to %s"%(local_zip, dataset_dir,))
        sys.stdout.flush()
        zf.extractall(path=dataset_dir)

      FLAGS.dataset = os.path.join(dataset_dir, 'img_align_celeba')

    filenames = tf.gfile.ListDirectory(FLAGS.dataset)
    filenames = [fn for fn in filenames if fn != '.floyddata']
    print("%d files in %s"%(len(filenames), FLAGS.dataset,))
    assert len(filenames) > 0
    sys.stdout.flush()
    filenames = sorted(filenames)
    random.shuffle(filenames)
    filenames = [os.path.join(FLAGS.dataset, f) for f in filenames]

    return filenames


def setup_tensorflow():
    # Create session
    config = tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement,
      allow_soft_placement=True,
    )
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

    return sess, summary_writer

def _(checkpoint_dir, input_image_filename):
    # Load checkpoint
    if not tf.gfile.IsDirectory(checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (checkpoint_dir,))

    # Setup global tensorflow state
    config = tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement,
      allow_soft_placement=True,
    )
    sess = tf.Session(config=config)

    # Initialize rng with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    # Create and initialize model
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, features, labels)

    feed_dict = {gene_minput: test_feature}
    gene_output = sess.run(td.gene_moutput, feed_dict=feed_dict)

    # replace test_fea
    _summarize_progress(td, test_feature, test_label, gene_output, batch,
                        'out', max_samples=1)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

def _demo():
    # Load checkpoint
    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.checkpoint_dir,))

    # Prepare directories
    filenames = prepare_dirs(delete_train_dir=False)

    # Setup global tensorflow state
    sess, summary_writer = setup_tensorflow()

    # Setup async input queues
    features, labels = srez_input.setup_inputs(sess, filenames)

    # Create and initialize model
    with tf.device(FLAGS.device):
      [gene_minput, gene_moutput,
       gene_output, gene_var_list,
       disc_real_output, disc_fake_output, disc_var_list] = \
              srez_model.create_model(sess, features, labels)

    # Restore variables from checkpoint
    saver = tf.train.Saver()
    filename = 'checkpoint_new.txt'
    filename = os.path.join(FLAGS.checkpoint_dir, filename)
    saver.restore(sess, filename)

    # Execute demo
    srez_demo.demo1(sess)

class TrainData(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _train():
  # Prepare directories
  all_filenames = prepare_dirs(delete_train_dir=True)

  # Setup global tensorflow state
  sess, summary_writer = setup_tensorflow()

  # Separate training and test sets
  train_filenames = all_filenames[:-FLAGS.test_vectors]
  test_filenames  = all_filenames[-FLAGS.test_vectors:]

  # TBD: Maybe download dataset here

  # Setup async input queues
  with tf.device("/cpu:0"):
    train_features, train_labels = srez_input.setup_inputs(sess, train_filenames)
    test_features,  test_labels  = srez_input.setup_inputs(sess, test_filenames)

    # Add some noise during training (think denoising autoencoders)
    noise_level = .03
    noisy_train_features = train_features + \
                         tf.random_normal(train_features.get_shape(), stddev=noise_level)

  # Create and initialize model
  with tf.device(FLAGS.device):
    [gene_minput, gene_moutput,
     gene_output, gene_var_list,
     disc_real_output, disc_fake_output, disc_var_list] = \
            srez_model.create_model(sess, noisy_train_features, train_labels)

    gene_loss = srez_model.create_generator_loss(disc_fake_output, gene_output, train_features)
    disc_real_loss, disc_fake_loss = \
                     srez_model.create_discriminator_loss(disc_real_output, disc_fake_output)
    disc_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')

    (global_step, learning_rate, gene_minimize, disc_minimize) = \
            srez_model.create_optimizers(gene_loss, gene_var_list,
                                         disc_loss, disc_var_list)

  # Train model
  train_data = TrainData(locals())
  srez_train.train_model(train_data)

def main(argv=None):
  # Training or showing off?
  if tf.__version__ != '1.0.0':
    print("please use tensorflow version 1.0.0")
    exit(0)
  if FLAGS.run == 'demo':
    #_demo()
    srez_demo.demo1()
  elif FLAGS.run == 'train':
    _train()

if __name__ == '__main__':
  tf.app.run()
