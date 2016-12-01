import tensorflow as tf
import numpy as np

import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--restore", help="restore previously trained weights (default=false)")
parser.add_argument("-s", "--strong_supervision", help="use labelled supporting facts (default=false)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
parser.add_argument("-l", "--l2_loss", type=float, default=0.001, help="specify l2 loss constant")
parser.add_argument("-n", "--num_runs", type=int, help="specify the number of model runs")

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

config.l2 = args.l2_loss if args.l2_loss is not None else 0.001
config.strong_supervision = args.strong_supervision if args.strong_supervision is not None else False
num_runs = args.num_runs if args.num_runs is not None else 1

print 'Training DMN ' + dmn_type + ' on snli data'

best_overall_val_loss = float('inf')

# create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        from dmn_original import DMN
        model = DMN(config)
    elif dmn_type == "plus":
        from dmn_plus import DMN_PLUS
        model = DMN_PLUS(config)

for run in range(num_runs):

    print 'Starting run', run

    print '==> initializing variables'
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:

        sum_dir = 'summaries/train/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.train.SummaryWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        if args.restore:
            print '==> restoring weights'
            saver.restore(session, 'weights/snli.weights')

        print '==> starting training'
        for epoch in xrange(config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()

            train_loss, train_accuracy = model.run_epoch(
              session, model.train, epoch, train_writer,
              train_op=model.train_step, train=True)
            valid_loss, valid_accuracy = model.run_epoch(session, model.valid)
            print 'Training loss: {}'.format(train_loss)
            print 'Validation loss: {}'.format(valid_loss)
            print 'Training accuracy: {}'.format(train_accuracy)
            print 'Vaildation accuracy: {}'.format(valid_accuracy)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print 'Saving weights'
                    best_overall_val_loss = best_val_loss
                    best_val_accuracy = valid_accuracy
                    saver.save(session, 'snli.weights')

            # anneal
            if train_loss>prev_epoch_loss*model.config.anneal_threshold:
                model.config.lr/=model.config.anneal_by
                print 'annealed lr to %f'%model.config.lr

            prev_epoch_loss = train_loss


            if epoch - best_val_epoch > config.early_stopping:
                break
            print 'Total time: {}'.format(time.time() - start)

        print 'Best validation accuracy:', best_val_accuracy



