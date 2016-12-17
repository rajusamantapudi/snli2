import tensorflow as tf
import numpy as np

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
args = parser.parse_args()

dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "original":
    from dmn_original import Config
    config = Config()
elif dmn_type == "plus":
    from dmn_plus import Config
    config = Config()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')


config.strong_supervision = False

config.train_mode = False

print 'Testing DMN ' + dmn_type

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        from dmn_original import DMN
        model = DMN(config)
    elif dmn_type == "plus":
        from dmn_plus import DMN_PLUS
        model = DMN_PLUS(config)

print '==> initializing variables'
init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print '==> restoring weights'
    saver.restore(session, 'snli.weights')

    print '==> running DMN'
    test_loss, test_accuracy = model.run_epoch(session, model.test)

    print ''
    print 'Test accuracy:', test_accuracy
