""" sentence to sentence training, evaluation, and inference. (compatible with TensorFlow v1.8)"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange  # pylint: disable=redefined-builtin

try:
    from ConfigParser import SafeConfigParser
except:
    # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.
    from configparser import SafeConfigParser 

import sys
import pdb

import tensorflow as tf

import data_utils
import seq2seq_model


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
# _buckets = [(15, 15), (25, 25), (35, 35), (45, 45), (60, 60)]

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    _conf_booleans = [ (name, parser.getboolean('booleans', name))
                        for name in parser.options('booleans') ]
    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_booleans)

def create_model(session, forward_only):

    """Create model and initialize or load parameters"""
    model = seq2seq_model.Seq2SeqModel( gConfig['enc_vocab_size'],
                                        gConfig['dec_vocab_size'], _buckets,
                                        gConfig['layer_size'], gConfig['num_layers'],
                                        gConfig['max_gradient_norm'],
                                        gConfig['batch_size'],
                                        gConfig['learning_rate'],
                                        gConfig['learning_rate_decay_factor'],
                                        forward_only=forward_only,
                                        use_pretrained_embedding=gConfig['use_pretrained_embedding'],
                                        pretrained_embedding_path=gConfig['pretrained_embedding_path'])
    pdb.set_trace()

    if 'pretrained_model' in gConfig:
          model.saver.restore(session,gConfig['pretrained_model'])
          return model

    ckpt = tf.train.get_checkpoint_state(gConfig['model_directory'])
    if ckpt and ckpt.model_checkpoint_path:
        #   print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #   model.saver.restore(session, ckpt.model_checkpoint_path)
          print("Reading parameters from previous model ...")
          model.saver.restore(session, "model/caption_300units_noatt_50kvocab/seq2seq.ckpt-625000")

    else:
          print("Created model with fresh parameters.")
          session.run(tf.global_variables_initializer())
    return model

def train():
    # prepare dataset
    print("Preparing data in %s" % gConfig['working_dir'])
    enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_custom_data(gConfig['working_dir'],gConfig['train_enc'],gConfig['train_dec'],gConfig['test_enc'],gConfig['test_dec'],gConfig['enc_vocab_size'],gConfig['dec_vocab_size'])
    
    # setup config to use BFC allocator
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['layer_size']))
        model = create_model(sess, False)


    # pdb.set_trace()



def decode():
    pass

def multi_test():
    pass

def scorer():
    pass

if __name__ == '__main__':

    if len(sys.argv) - 1:
        gConfig = get_config(sys.argv[1])
    else:
        # get configuration from seq2seq.ini
        gConfig = get_config()

    if not tf.gfile.Exists(gConfig['model_directory']):
        tf.gfile.MakeDirs(gConfig['model_directory'])
    if not tf.gfile.Exists(gConfig['log_dir']):
        tf.gfile.MakeDirs(gConfig['log_dir'])
    if not tf.gfile.Exists(gConfig['result_dir']):
        tf.gfile.MakeDirs(gConfig['result_dir'])

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        # start training
        train()
    elif gConfig['mode'] == 'test':
        # interactive decode
        decode()
    elif gConfig['mode'] == 'eval':
        scorer()
    elif gConfig['mode'] == 'generate':
        multi_test()
    else:
        # wrong way to execute "serve"
        #   Use : >> python ui/app.py
        #           uses seq2seq_serve.ini as conf file
        print('Serve Usage : >> python ui/app.py')
        print('# uses seq2seq_serve.ini as conf file')        

# pdb.set_trace()
