"""
Training script for permute MNIST experiment.
"""
from __future__ import print_function
import collections
import argparse
import os
import sys
import math
import time

import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from six.moves import cPickle as pickle
from tqdm import tqdm

from utils.data_utils import construct_permute_mnist
from utils.utils import get_sample_weights, sample_from_dataset, update_episodic_memory, concatenate_datasets, \
    samples_for_each_class, sample_from_dataset_icarl, compute_fgt, update_reservior
from utils.vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, \
    snapshot_experiment_eval
from model import Model

###############################################################
################ Some definitions #############################
### These will be edited by the command line options ##########
###############################################################

## Training Options
NUM_RUNS = 10  # Number of experiments to average over
TRAIN_ITERS = 5000  # Number of training iterations per task
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
RANDOM_SEED = 1234  # 1234
VALID_OPTIMS = ['SGD', 'MOMENTUM', 'ADAM']
OPTIM = 'SGD'
OPT_POWER = 0.9
OPT_MOMENTUM = 0.9
VALID_ARCHS = ['FC-S', 'FC-B']
ARCH = 'FC-S'

alpha = 0.1

## Model options
MODELS = ['VAN', 'PI', 'EWC', 'MAS', 'RWALK', 'A-GEM', 'S-GEM', 'FTR_EXT', 'PNN', 'ER', 'MEGA', 'MEGAD', 'MEGA_RA',
          'AKCL']  # List of valid models
IMP_METHOD = 'EWC'
# SYNAP_STGTH = 75000
SYNAP_STGTH = 100000000
FISHER_EMA_DECAY = 0.9  # Exponential moving average decay factor for Fisher computation (online Fisher)
FISHER_UPDATE_AFTER = 10  # Number of training iterations for which the F_{\theta}^t is computed (see Eq. 10 in RWalk paper)

SAMPLES_PER_CLASS = 25  # Number of samples per task

INPUT_FEATURE_SIZE = 784
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1
TOTAL_CLASSES = 10  # Total number of classes in the dataset

EPS_MEM_BATCH_SIZE = 256

DEBUG_EPISODIC_MEMORY = False
USE_GPU = True
K_FOR_CROSS_VAL = 3

TIME_MY_METHOD = False
COUNT_VIOLATIONS = False
MEASURE_PERF_ON_EPS_MEMORY = False

## Logging, saving and testing options
LOG_DIR = './permute_mnist_results'

## Evaluation options

## Num Tasks
NUM_TASKS = 20

MULTI_TASK = False


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for permutted mnist experiment.")
    parser.add_argument("--cross-validate-mode", action="store_true",
                        help="If option is chosen then snapshoting after each batch is disabled")
    parser.add_argument("--online-cross-val", action="store_true",
                        help="If option is chosen then enable the online cross validation of the learning rate")
    parser.add_argument("--train-single-epoch", action="store_true",
                        help="If option is chosen then train for single epoch")
    parser.add_argument("--eval-single-head", action="store_true",
                        help="If option is chosen then evaluate on a single head setting.")
    parser.add_argument("--arch", type=str, default=ARCH, help="Network Architecture for the experiment.\
                        \n \nSupported values: %s" % (VALID_ARCHS))
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS,
                        help="Total runs/ experiments over which accuracy is averaged.")
    parser.add_argument("--train-iters", type=int, default=TRAIN_ITERS,
                        help="Number of training iterations for each task.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Mini-batch size for each task.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random Seed.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Starting Learning rate for each task.")
    parser.add_argument("--optim", type=str, default=OPTIM,
                        help="Optimizer for the experiment. \
                                \n \nSupported values: %s" % (VALID_OPTIMS))
    parser.add_argument("--imp-method", type=str, default=IMP_METHOD,
                        help="Model to be used for LLL. \
                        \n \nSupported values: %s" % (MODELS))
    parser.add_argument("--synap-stgth", type=float, default=SYNAP_STGTH,
                        help="Synaptic strength for the regularization.")
    parser.add_argument("--fisher-ema-decay", type=float, default=FISHER_EMA_DECAY,
                        help="Exponential moving average decay for Fisher calculation at each step.")
    parser.add_argument("--fisher-update-after", type=int, default=FISHER_UPDATE_AFTER,
                        help="Number of training iterations after which the Fisher will be updated.")
    parser.add_argument("--mem-size", type=int, default=SAMPLES_PER_CLASS,
                        help="Number of samples per class from previous tasks.")
    parser.add_argument("--eps-mem-batch", type=int, default=EPS_MEM_BATCH_SIZE,
                        help="Number of samples per class from previous tasks.")
    parser.add_argument("--examples-per-task", type=int, default=1000,
                        help="Number of examples per task.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Directory where the plots and model accuracies will be stored.")
    parser.add_argument("--m1-value", type=float, default=0.4)
    parser.add_argument("--m2-value", type=float, default=0.01)
    parser.add_argument("--s-value", type=int, default=24)
    return parser.parse_args()


MFR_matrix = np.zeros([17, 17, 10, 256])
TSNE_feature = []
TSNE_label = []
MFR_list = []


def train_task_sequence(model, sess, args):
    """
    Train and evaluate LLL system such that we only see a example once
    Args:
    Returns:
        dict    A dictionary containing mean and stds for the experiment
    """
    # List to store accuracy for each run
    runs = []

    batch_size = args.batch_size

    if model.imp_method == 'A-GEM' or model.imp_method == 'ER' or model.imp_method == 'MEGA' or model.imp_method == 'MEGAD' or model.imp_method == 'MEGA_RA' or model.imp_method == 'AKCL':
        use_episodic_memory = True
    else:
        use_episodic_memory = False

    # Loop over number of runs to average over
    for runid in range(args.num_runs):
        print('\t\tRun %d:' % (runid))

        # Initialize the random seeds
        np.random.seed(args.random_seed + runid)

        # Load the permute mnist dataset
        datasets = construct_permute_mnist(model.num_tasks)

        episodic_mem_size = args.mem_size * model.num_tasks * TOTAL_CLASSES
        episodic_mem_size_per_task = args.mem_size * TOTAL_CLASSES
        print("buffer size")
        print(episodic_mem_size)
        # Initialize all the variables in the model
        sess.run(tf.global_variables_initializer())

        # Run the init ops
        model.init_updates(sess)

        # List to store accuracies for a run
        evals = []

        # List to store the classes that we have so far - used at test time
        test_labels = np.arange(TOTAL_CLASSES)

        if use_episodic_memory:
            # Reserve a space for episodic memory
            episodic_images = np.zeros([episodic_mem_size, INPUT_FEATURE_SIZE])
            episodic_labels = np.zeros([episodic_mem_size, TOTAL_CLASSES])
            episodic_images_tasks = [np.zeros([episodic_mem_size_per_task, INPUT_FEATURE_SIZE]) for i in
                                     range(model.num_tasks)]
            episodic_labels_tasks = [np.zeros([episodic_mem_size_per_task, TOTAL_CLASSES]) for i in
                                     range(model.num_tasks)]

            episodic_features = np.zeros([episodic_mem_size, 256])

            count_cls = np.zeros(TOTAL_CLASSES, dtype=np.int32)
            episodic_filled_counter = 0
            examples_seen_so_far = 0

        # Mask for softmax
        # Since all the classes are present in all the tasks so nothing to mask
        logit_mask = np.ones(TOTAL_CLASSES)
        if model.imp_method == 'PNN':
            pnn_train_phase = np.array(np.zeros(model.num_tasks), dtype=np.bool)
            pnn_logit_mask = np.ones([model.num_tasks, TOTAL_CLASSES])

        if COUNT_VIOLATIONS:
            violation_count = np.zeros(model.num_tasks)
            vc = 0

        # Training loop for all the tasks
        feat_mean = {}
        feat_var = {}
        feat_means_angle = {}

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        tsne = TSNE(n_components=2, random_state=0)

        cur_task_grads = []
        cur_task_ids = []
        grads_deviation = []
        grads_angles = []
        grads_cur = []
        grads_ref = []
        grads_ratio = []
        loss_cur = []
        loss_ref = []

        for task in range(len(datasets)):
            this_cur_task_grads = []
            this_cur_task_ids = []
            print('\t%s\tTask %d:' % (model.imp_method, task))

            # If not the first task then restore weights from previous task
            if (task > 0 and model.imp_method != 'PNN'):
                model.restore(sess)

            # Extract training images and labels for the current task
            task_train_images = datasets[task]['train']['images']
            task_train_labels = datasets[task]['train']['labels']

            # If multi_task is set the train using datasets of all the tasks
            if MULTI_TASK:
                if task == 0:
                    for t_ in range(1, len(datasets)):
                        task_train_images = np.concatenate((task_train_images, datasets[t_]['train']['images']), axis=0)
                        task_train_labels = np.concatenate((task_train_labels, datasets[t_]['train']['labels']), axis=0)
                else:
                    # Skip training for this task
                    continue

            # Assign equal weights to all the examples
            task_sample_weights = np.ones([task_train_labels.shape[0]], dtype=np.float32)
            total_train_examples = task_train_images.shape[0]
            # Randomly suffle the training examples
            perm = np.arange(total_train_examples)
            np.random.shuffle(perm)
            train_x = task_train_images[perm]  # [:examples_per_task]
            train_y = task_train_labels[perm]  # [:examples_per_task]
            task_sample_weights = task_sample_weights[perm]  # [:examples_per_task]

            print('Received {} images, {} labels at task {}'.format(train_x.shape[0], train_y.shape[0], task))

            # Array to store accuracies when training for task T
            ftask = []

            num_train_examples = train_x.shape[0]

            # Train a task observing sequence of data
            if args.train_single_epoch:
                num_iters = num_train_examples // batch_size
            else:
                num_iters = args.train_iters

            # Training loop for task T
            # print (num_iters)

            stored_grads = np.zeros([model.num_tasks, 269322], dtype=np.float)
            stored_ref_loss = np.zeros([model.num_tasks], dtype=np.float)
            for iters in tqdm(range(num_iters)):

                # if iters == 2750:
                #     fbatch, flogits, fy, theta = test_task_sequence(model, sess, datasets, task, False)
                #     ys = np.argmax(fy[0], axis=-1)[:]
                #     theta_1 = theta[0]
                #     print(theta_1)
                #     print(theta_1[0])
                #     theta_pred = {}
                #     print('a')
                #     for i in range(len(theta_1)):
                #         idx_t = ys[i]
                #         if ys[i] not in theta_pred.keys():
                #             theta_pred[ys[i]] = [theta_1[i][idx_t]]
                #         else:
                #             theta_pred[ys[i]].append(theta_1[i][idx_t])
                #     f_t = open("theta/theta_2750.txt", 'w')
                #     f_t.write(str(theta_pred))
                #     f_t.close()
                #     exit()

                # if args.train_single_epoch and not args.cross_validate_mode:
                #     if (iters < 10) or (iters < 100 and iters % 10 == 0) or (iters % 100 == 0):
                #         # Snapshot the current performance across all tasks after each mini-batch
                #         time1 = time.time()
                #         fbatch, flogits, fy = test_task_sequence(model, sess, datasets, task, args.online_cross_val)
                #         time2 = time.time()
                #         print('test time: {}'.format(time2-time1))
                #         with open('time.txt', 'a') as f:
                #             f.write('mnist_m1_{}_m2_{}_s_{}_method_{}_test_time:{}'.format(args.m1_value, args.m2_value,
                #                                                                      args.s_value, args.imp_method,
                #                                                                      time2-time1))
                #             f.write('\n')
                #         if iters == 5:
                #             exit()
                #         ftask.append(fbatch)

                offset = (iters * batch_size) % (num_train_examples - batch_size)
                residual = batch_size

                if model.imp_method == 'PNN':
                    pnn_train_phase[:] = False
                    pnn_train_phase[task] = True
                    feed_dict = {model.x: train_x[offset:offset + batch_size],
                                 model.y_[task]: train_y[offset:offset + batch_size],
                                 model.sample_weights: task_sample_weights[offset:offset + batch_size],
                                 model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0}
                    train_phase_dict = {m_t: i_t for (m_t, i_t) in zip(model.train_phase, pnn_train_phase)}
                    logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, pnn_logit_mask)}
                    feed_dict.update(train_phase_dict)
                    feed_dict.update(logit_mask_dict)
                else:
                    feed_dict = {model.x: train_x[offset:offset + batch_size],
                                 model.y_: train_y[offset:offset + batch_size],
                                 model.sample_weights: task_sample_weights[offset:offset + batch_size],
                                 model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                 model.output_mask: logit_mask, model.train_phase: True}

                if model.imp_method == 'VAN':
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PNN':
                    feed_dict[model.task_id] = task
                    _, loss = sess.run([model.train[task], model.unweighted_entropy[task]], feed_dict=feed_dict)

                elif model.imp_method == 'FTR_EXT':
                    if task == 0:
                        _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                    else:
                        _, loss = sess.run([model.train_classifier, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'EWC':
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                    # Update fisher after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        sess.run(model.set_running_fisher)
                        sess.run(model.reset_tmp_fisher)

                    _, _, loss = sess.run([model.set_tmp_fisher, model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PI':
                    _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.train, model.update_small_omega,
                                              model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'MAS':
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'A-GEM':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch,
                                                               replace=False)  # Sample without replacement so that we don't sample an example more than once

                        # Store the reference gradient
                        sess.run(model.store_ref_grads, feed_dict={model.x: episodic_images[mem_sample_mask],
                                                                   model.y_: episodic_labels[mem_sample_mask],
                                                                   model.keep_prob: 1.0, model.output_mask: logit_mask,
                                                                   model.train_phase: True})

                        if COUNT_VIOLATIONS:
                            vc, _, loss = sess.run([model.violation_count, model.train_subseq_tasks, model.agem_loss],
                                                   feed_dict=feed_dict)
                        else:
                            # Compute the gradient for current task and project if need be
                            _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)

                    # if True and task in [1]:
                    #     norm_cur_grads = np.linalg.norm(cur_grads, keepdims=False)
                    #     norm_ref_grads = np.linalg.norm(ref_grads, keepdims=False)
                    #     grads_cur.append(norm_cur_grads)
                    #     grads_ref.append(norm_ref_grads)
                    #     grads_show = pd.DataFrame({'cur':grads_cur, 'ref':grads_ref})
                    #     grads_deviation.append(norm_cur_grads-norm_ref_grads)
                    #     angle = np.dot(cur_grads, ref_grads)
                    #     angle = np.arccos(angle/(norm_cur_grads*norm_ref_grads))
                    #     angle = (angle/np.pi)*180
                    #     grads_angles.append(angle)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = cls
                        with_in_task_offset = args.mem_size * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size

                elif model.imp_method == 'MEGA':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:

                        ## Compute and store the reference gradients on the previous tasks
                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch,
                                                               replace=False)  # Sample without replacement so that we don't sample an example more than once

                        # Store the reference gradient
                        _, ref_loss = sess.run([model.store_ref_grads, model.store_ref_loss],
                                               feed_dict={model.x: episodic_images[mem_sample_mask],
                                                          model.y_: episodic_labels[mem_sample_mask],
                                                          model.keep_prob: 1.0, model.output_mask: logit_mask,
                                                          model.train_phase: True})

                        # Compute the gradient for current task and project if need be
                        _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)

                    # # if True and task in [0]:
                    # #     cur_task_grads.append(cur_grads)
                    # #     cur_task_ids.append('0-0')
                    # #     # cur_task_grads.append(ref_grads)
                    # #     # cur_task_ids.append(0)
                    # if True and task in [1]:
                    #     norm_cur_grads = np.linalg.norm(cur_grads, keepdims=False)
                    #     norm_ref_grads = np.linalg.norm(ref_grads, keepdims=False)
                    #     grads_cur.append(norm_cur_grads)
                    #     grads_ref.append(norm_ref_grads)
                    #     grads_show = pd.DataFrame({'cur':grads_cur, 'ref':grads_ref})
                    #     grads_deviation.append(norm_cur_grads-norm_ref_grads)
                    #     angle = np.dot(cur_grads, ref_grads)
                    #     angle = np.arccos(angle/(norm_cur_grads*norm_ref_grads))
                    #     angle = (angle/np.pi)*180
                    #     # print(angle)
                    #     # print(angle/(norm_cur_grads*norm_ref_grads))
                    #     # print(np.arccos(angle/(norm_cur_grads*norm_ref_grads)))
                    #     # exit()
                    #     grads_angles.append(angle)
                    #     # print(norm_cur_grads)
                    #     # print(norm_ref_grads)
                    #     # print(norm_cur_grads-norm_ref_grads)
                    #     # exit()
                    #     # cur_task_grads.append(cur_grads)
                    #     # cur_task_ids.append('1-1')
                    #     # cur_task_grads.append(ref_grads)
                    #     # cur_task_ids.append('1-0')
                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = cls
                        with_in_task_offset = args.mem_size * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size
                elif model.imp_method == 'MEGAD':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:

                        ## Compute and store the reference gradients on the previous tasks
                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch,
                                                               replace=False)  # Sample without replacement so that we don't sample an example more than once

                        # Store the reference gradient
                        _, ref_loss = sess.run([model.store_ref_grads, model.store_ref_loss],
                                               feed_dict={model.x: episodic_images[mem_sample_mask],
                                                          model.y_: episodic_labels[mem_sample_mask],
                                                          model.keep_prob: 1.0, model.output_mask: logit_mask,
                                                          model.train_phase: True})

                        # Compute the gradient for current task and project if need be
                        _, loss, ratio = sess.run([model.train_subseq_tasks, model.agem_loss, model.ratio],
                                                  feed_dict=feed_dict)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = cls
                        with_in_task_offset = args.mem_size * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size

                elif model.imp_method == 'AKCL':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                        # _, loss, theta = sess.run([model.train_first_task, model.agem_loss, model.thetaww],
                        #                           feed_dict=feed_dict)
                        # print(theta)
                        # exit()
                    else:

                        ## Compute and store the reference gradients on the previous tasks
                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch,
                                                               replace=False)  # Sample without replacement so that we don't sample an example more than once

                        # Store the reference gradient
                        ref_loss, ref_grads, kl_grads = sess.run(
                            [model.store_ref_loss, model.ref_grads_save, model.kl_grads_save], feed_dict={
                                model.x: episodic_images[mem_sample_mask],
                                model.y_: episodic_labels[mem_sample_mask],
                                model.org_feat: episodic_features[mem_sample_mask],
                                model.flag1: 1,
                                model.keep_prob: 1.0,
                                model.output_mask: logit_mask,
                                model.train_phase: True})

                        stored_grads = ref_grads
                        stored_loss = ref_loss
                        stored_kl_grads = kl_grads

                        # Compute the gradient for current task and project if need be
                        feed_dict[model.store_grads] = stored_grads
                        feed_dict[model.store_kl_grads] = stored_kl_grads
                        feed_dict[model.store_loss] = [stored_loss]
                        feed_dict[model.flag1] = 0
                        _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = cls
                        with_in_task_offset = args.mem_size * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size
                        cur_feat = sess.run(model.features,
                                            feed_dict={
                                                model.x: np.array([episodic_images[mem_index]]),
                                                model.y_: np.array([episodic_labels[mem_index]]),
                                                model.flag1: 0,
                                                model.keep_prob: 1.0,
                                                model.train_phase: False})
                        # print(cur_feat.shape)
                        episodic_features[mem_index] = cur_feat

                elif model.imp_method == 'MEGA_RA':
                    if task == 0:
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        for k in range(task):
                            # Here, we need to keep the covariance
                            # 1. for each class, we 
                            # x = episodic_images_tasks[k]
                            # y = episodic_labels_tasks[k]
                            # aug_eps_image = []
                            # aug_eps_label = []
                            # for x, y in zip(episodic_images_tasks[k], episodic_labels_tasks[k]):
                            #     mean = feat_mean[k][np.unique(np.nonzero(y))[-1]]
                            #     var = feat_var[k][np.unique(np.nonzero(y))[-1]]
                            #     # starttime = datetime.datetime.now()
                            #     x_tilde = np.random.multivariate_normal(np.zeros([10]), .1*var)
                            #     # endtime = datetime.datetime.now()
                            #     # print(endtime - starttime)
                            #     aug_eps_image.append(x_tilde)
                            #     # aug_eps_label.append(y)

                            ref_grads, ref_loss = sess.run([model.ref_grads_save, model.agem_loss],
                                                           feed_dict={
                                                               model.x: episodic_images_tasks[k],
                                                               model.y_: episodic_labels_tasks[k],
                                                               model.keep_prob: 1.0,
                                                               model.output_mask: logit_mask,
                                                               model.train_phase: True,
                                                               model.current_task_id: task,
                                                               # model.reset_storage:False,
                                                               model.ref_loss: stored_ref_loss[k]})
                            stored_ref_loss[k] = ref_loss
                            stored_grads[k] = ref_grads

                        feed_dict = {model.x: train_x[offset:offset + batch_size],
                                     model.y_: train_y[offset:offset + batch_size],
                                     model.sample_weights: task_sample_weights[offset:offset + batch_size],
                                     model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                     model.output_mask: logit_mask, model.train_phase: True,
                                     model.current_task_id: task,
                                     model.store_grads: stored_grads[:task]}
                        _, loss, ref_grad = sess.run([model.train_subseq_tasks, model.agem_loss, model.ref_grads],
                                                     feed_dict=feed_dict)

                    # if True and task in [1]:
                    #     norm_cur_grads = np.linalg.norm(cur_grads, keepdims=False)
                    #     norm_ref_grads = np.linalg.norm(ref_grads, keepdims=False)
                    #     grads_cur.append(norm_cur_grads)
                    #     grads_ref.append(norm_ref_grads)
                    #     grads_show = pd.DataFrame({'cur':grads_cur, 'ref':grads_ref})
                    #     grads_deviation.append(norm_cur_grads-norm_ref_grads)
                    #     angle = np.dot(cur_grads, ref_grads)
                    #     angle = np.arccos(angle/(norm_cur_grads*norm_ref_grads))
                    #     angle = (angle/np.pi)*180
                    #     grads_angles.append(angle)
                    #     grads_ratio.append(ratio)
                    #     loss_cur.append(loss)
                    #     loss_ref.append(ref_loss)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = cls
                        with_in_task_offset = args.mem_size * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size

                        episodic_images_tasks[task][count_cls[cls] + with_in_task_offset] = er_x
                        episodic_labels_tasks[task][count_cls[cls] + with_in_task_offset] = er_y_


                elif model.imp_method == 'RWALK':
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                    # Update fisher and importance score after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        # Update the importance score using distance in riemannian manifold   
                        sess.run(model.update_big_omega_riemann)
                        # Now that the score is updated, compute the new value for running Fisher
                        sess.run(model.set_running_fisher)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                        # Reset the delta_L
                        sess.run([model.reset_small_omega])

                    _, _, _, _, loss = sess.run([model.set_tmp_fisher, model.weights_old_ops_grouped,
                                                 model.train, model.update_small_omega, model.reg_loss],
                                                feed_dict=feed_dict)

                elif model.imp_method == 'ER':
                    mem_filled_so_far = examples_seen_so_far if (
                            examples_seen_so_far < episodic_mem_size) else episodic_mem_size
                    if mem_filled_so_far < args.eps_mem_batch:
                        er_mem_indices = np.arange(mem_filled_so_far)
                    else:
                        er_mem_indices = np.random.choice(mem_filled_so_far, args.eps_mem_batch, replace=False)

                    np.random.shuffle(er_mem_indices)
                    # Train on a batch of episodic memory first
                    er_train_x_batch = np.concatenate(
                        (episodic_images[er_mem_indices], train_x[offset:offset + residual]), axis=0)
                    er_train_y_batch = np.concatenate(
                        (episodic_labels[er_mem_indices], train_y[offset:offset + residual]), axis=0)

                    feed_dict = {model.x: er_train_x_batch, model.y_: er_train_y_batch,
                                 model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 1.0,
                                 model.output_mask: logit_mask, model.train_phase: True}
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        update_reservior(er_x, er_y_, episodic_images, episodic_labels, episodic_mem_size,
                                         examples_seen_so_far)
                        examples_seen_so_far += 1

                # if (iters % 100 == 0):
                #     print('Step {:d} {:.3f}'.format(iters, loss))

                if (math.isnan(loss)):
                    print('ERROR: NaNs NaNs Nans!!!')
                    sys.exit(0)

            print('\t\t\t\tTraining for Task%d done!' % (task))
            if False and task in [1]:
                # sns.distplot(grads_deviation)
                # sns.distplot(grads_angles)
                # sns.lineplot(list(range(len(grads_deviation))), grads_angles)
                # sns.barplot(list(range(len(grads_deviation))), grads_deviation)

                fig, axes = plt.subplots(2, 5)
                sns.distplot(grads_cur, ax=axes[0, 0])
                # # print(grads_ref)
                sns.distplot(grads_ref, ax=axes[0, 1])
                # sns.distplot(grads_ratio)
                sns.distplot(loss_cur, ax=axes[0, 2])
                sns.distplot(loss_ref, ax=axes[0, 3])
                sns.distplot(grads_ratio, ax=axes[0, 4])
                sns.distplot(grads_deviation, ax=axes[1, 0])
                sns.distplot(grads_angles, ax=axes[1, 1])
                sns.lineplot(list(range(len(grads_ratio))), grads_ratio, ax=axes[1, 4])

                plt.savefig('./tsne_plot/{}/grads_ratio25-{}.pdf'.format(model.imp_method, task))
                plt.close()
                exit()

            if False:
                # save class mean and covariance
                feat_mean[task] = {}
                feat_var[task] = {}
                # compute mean
                for er_x, er_y in tqdm(zip(train_x, train_y)):
                    y = np.unique(np.nonzero(er_y))[-1]
                    if y not in feat_mean[task].keys():
                        feat_mean[task][y] = 0.
                    feat_x = sess.run(model.out_logits, feed_dict={
                        model.x: [er_x],
                        model.y_: [er_y],
                        model.keep_prob: 1.0,
                        model.train_phase: False})
                    feat_mean[task][y] += feat_x[0]

                means = np.zeros([10, 10])
                for k, v in feat_mean[task].items():
                    v /= 5500.
                    means[k] = v

                means_norm = np.linalg.norm(means, axis=1, keepdims=True)
                feat_means_angle[task] = np.matmul(means, np.transpose(means)) / np.matmul(means_norm,
                                                                                           np.transpose(means_norm))

                # compute covariance
                for er_x, er_y in tqdm(zip(train_x, train_y)):
                    y = np.unique(np.nonzero(er_y))[-1]
                    if y not in feat_var[task].keys():
                        feat_var[task][y] = 0.
                    feat_x = sess.run(model.out_logits, feed_dict={
                        model.x: [er_x],
                        model.y_: [er_y],
                        model.keep_prob: 1.0,
                        model.train_phase: False})

                    a = feat_x - feat_mean[task][y]
                    feat_var[task][y] += np.matmul(np.transpose(a, [1, 0]), a)

                for k, v in feat_var[task].items():
                    v /= 5500.

            ########################################

            # Upaate the episodic memory filled counter
            if use_episodic_memory:
                episodic_filled_counter += args.mem_size * TOTAL_CLASSES

            if model.imp_method == 'A-GEM' and COUNT_VIOLATIONS:
                violation_count[task] = vc
                print('Task {}: Violation Count: {}'.format(task, violation_count))
                sess.run(model.reset_violation_count, feed_dict=feed_dict)

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if (task < (len(datasets) - 1)) or MEASURE_PERF_ON_EPS_MEMORY:
                model.task_updates(sess, task, task_train_images, np.arange(TOTAL_CLASSES))
                print('\t\t\t\tTask updates after Task%d done!' % (task))

            if args.train_single_epoch and not args.cross_validate_mode:
                fbatch, flogits, fy, theta = test_task_sequence(model, sess, datasets, task, False)

                # for j, (feature, yp) in enumerate(zip(flogits, fy)):
                #     if j <= task:
                #         yp = np.argmax(yp, axis=-1)[:]
                #         # print(yp) # 0-9
                #         # print(yp.shape) # 10000
                #         tmp_sum = np.zeros(10)
                #         for w, p in enumerate(yp):
                #             MFR_matrix[task][j][p] += feature[w]
                #             tmp_sum[p] += 1
                #         # print(MFR_matrix[task][j][p])
                #         # print(MFR_matrix[task][j][p].shape)
                #         for p in range(10):
                #             MFR_matrix[task][j][p] /= tmp_sum[p]
                #
                # # MFR_f_feature
                # if task == 16:
                #     MFR_matrix[np.isnan(MFR_matrix)] = 0
                #     mrf_score = 0
                #     for j in range(0, 16):
                #         tmp_l = 0
                #         for i in range(j + 1, 17):
                #             tmp_f = 0
                #             for c in range(0, 10):  # class num
                #                 tmp_f += np.linalg.norm(MFR_matrix[i][j][c] - MFR_matrix[j][j][c])
                #             tmp_l += tmp_f / 10 # class num
                #         mrf_score += tmp_l / (16 - j)
                #     mrf_score = mrf_score / 16
                #     MFR_list.append(mrf_score)
                #     print('MRF_score:{}!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(mrf_score))
                #     if runid == 4:
                #         with open('ab1_cifar.txt', 'a') as f:
                #             f.write('mnist_m1_{}_m2_{}_s_{}_method{}:'.format(args.m1_value, args.m2_value,
                #                                                               args.s_value,
                #                                                               args.imp_method))
                #             f.write(','.join(str(mfr_s) for mfr_s in MFR_list))
                #             f.write('\n')
                #             f.write('{},{}'.format(np.array(MFR_list).mean(), np.array(MFR_list).std()))
                #             f.write('\n')
                #
                # # MFR-r-SCORE
                # # if task == 16:
                # #     MFR_r = np.zeros([17, 17, 10, 10])
                # #     for a in range(17):
                # #         for b in range(a):
                # #             for c in range(10):
                # #                 for d in range(10):
                # #                     MFR_r[a][b][c][d] = np.dot(MFR_matrix[a][b][c].T, MFR_matrix[a][b][d]) / (
                # #                             np.linalg.norm(MFR_matrix[a][b][c]) * np.linalg.norm(MFR_matrix[a][b][d]))
                # #
                # #     mrf_score = 0
                # #     for j in range(0, 16):
                # #         tmp_l = 0
                # #         for i in range(j + 1, 17):
                # #             tmp_l += np.linalg.norm(MFR_r[i][j] - MFR_r[j][j])
                # #         tmp_l = tmp_l / (16 - j)
                # #         mrf_score += tmp_l
                # #     mrf_score = mrf_score / 16
                # #     with open('mrf_score.txt', 'a') as f:
                # #         f.write('mnist_m1_{}_m2_{}_s_{}_runid_{}:{}'.format(args.m1_value, args.m2_value,
                # #                                                           args.s_value, runid, mrf_score))
                # #         f.write('\n')
                # #     print('MRF_score:{}!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(mrf_score))

                ftask.append(fbatch)
                ftask = np.array(ftask)

                # <editor-fold desc='tsne-2020-07-18-1  task1 bianhua'>
                # TSNE code  tsne-2020-07-18-1  task1 bianhua
                if True and task in [0, 8, 16]:
                    if task == 0:
                        TSNE_feature = flogits[0].tolist()
                        print(len(flogits))
                        print(len(flogits[0]))
                        print(np.array(flogits).shape)
                        ys = np.argmax(fy[0], axis=-1)[:]
                        print(len(fy))
                        print(len(fy[0]))
                        print(ys.shape)
                        TSNE_label = ys.astype(int).tolist()
                    else:
                        TSNE_feature = TSNE_feature + flogits[0].tolist()
                        ys = np.argmax(fy[0], axis=-1)[:]
                        ys = ys + (task + 1) * 100
                        TSNE_label = TSNE_label + ys.astype(int).tolist()

                    if task == 16:
                        from sklearn.manifold import TSNE
                        import matplotlib.pyplot as plt
                        import pandas as pd
                        import seaborn as sns
                        # sns.set(font_scale=1.8)
                        tsne = TSNE(n_components=2, random_state=0)

                        print('TSNE for task {}-{}'.format(task, 0))
                        # # color1 = ['black', 'lightcoral', 'green', 'turquoise', 'darkorange']
                        # # color2 = ['black', 'lightcoral', 'green', 'turquoise', 'darkorange']
                        # # color3 = ['black', 'lightcoral', 'green', 'turquoise', 'darkorange']
                        # # color1 = ['indianred', 'orange', 'springgreen', 'lightblue', 'palevioletred']
                        # color1 = ['black', 'indianred', 'orange', 'springgreen', 'lightblue']
                        # color2 = ['black', 'indianred', 'orange', 'springgreen', 'lightblue']
                        # color3 = ['black', 'indianred', 'orange', 'springgreen', 'lightblue']
                        # color_a = color1 + color2 + color3
                        color1 = ['black', 'gray', 'brown', 'red', 'gold',
                                  'green', 'teal', 'blue', 'purple', 'fuchsia']  # pick
                        color_a = color1 + color1 + color1
                        markers = {"50": "o", "66": "o", "74": "o", "82": "o", "86": "o",
                                   "950": "x", "966": "x", "974": "x", "982": "x", "986": "x",
                                   "1750": "l", "1766": "l", "1774": "l", "1782": "l", "1786": "l"}

                        tsne_obj = tsne.fit_transform(TSNE_feature[:])

                        tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                                                'Y': tsne_obj[:, 1],
                                                'digit': np.array(TSNE_label)})
                        # tsne_df.head()
                        # print(tsne_df)
                        # print(tsne_df.head())
                        g = sns.scatterplot(x="X", y="Y",
                                            hue="digit",
                                            style="digit",
                                            palette=color_a[:10],
                                            legend='full',  # `legend` must be 'brief', 'full', or False
                                            markers=["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
                                            s=20,  # s=16 is equal with markersize=4
                                            # markers=markers,
                                            data=tsne_df.iloc[:10000, ])

                        plt.xlabel('')
                        plt.ylabel('')
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                        plt.xlim(-100, 100)
                        plt.ylim(-100, 100)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
                        # plt.savefig('./tsne_plot/compare_{}.png'.format(model.imp_method), dpi=300, bbox_inches='tight')
                        plt.savefig('./tsne_plot/m2_0.2/2_scale_compare_{}_runid_{}_task0.png'.format(model.imp_method, runid),
                                    dpi=600, bbox_inches='tight')
                        plt.close()
                        g = sns.scatterplot(x="X", y="Y",
                                            hue="digit",
                                            style="digit",
                                            palette=color_a[:10],
                                            legend='full',  # `legend` must be 'brief', 'full', or False
                                            markers=["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
                                            s=20,  # s=16 is equal with markersize=4
                                            # markers=markers,
                                            data=tsne_df.iloc[10000:20000, ])

                        plt.xlabel('')
                        plt.ylabel('')
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                        plt.xlim(-100, 100)
                        plt.ylim(-100, 100)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
                        # plt.savefig('./tsne_plot/compare_{}.png'.format(model.imp_method), dpi=300, bbox_inches='tight')
                        plt.savefig('./tsne_plot/m2_0.2/2_scale_compare_{}_runid_{}_task8.png'.format(model.imp_method, runid),
                                    dpi=600, bbox_inches='tight')
                        plt.close()
                        g = sns.scatterplot(x="X", y="Y",
                                            hue="digit",
                                            style="digit",
                                            palette=color_a[:10],
                                            legend='full',  # `legend` must be 'brief', 'full', or False
                                            markers=["o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
                                            s=20,  # s=16 is equal with markersize=4
                                            # markers=markers,
                                            data=tsne_df.iloc[20000:, ])

                        plt.xlabel('')
                        plt.ylabel('')
                        plt.xticks(fontsize=18)
                        plt.yticks(fontsize=18)
                        plt.xlim(-100, 100)
                        plt.ylim(-100, 100)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
                        # plt.savefig('./tsne_plot/compare_{}.png'.format(model.imp_method), dpi=300, bbox_inches='tight')
                        plt.savefig('./tsne_plot/m2_0.2/2_scale_compare_{}_runid_{}_task16.png'.format(model.imp_method, runid),
                                    dpi=600, bbox_inches='tight')
                        plt.close()
                        # g = sns.scatterplot(x="X", y="Y",
                        #                     hue="digit",
                        #                     style="digit",
                        #                     palette=color_a,
                        #                     legend='full',  # `legend` must be 'brief', 'full', or False
                        #                     markers=["s", "s", "s", "s", "s", "s", "s", "s", "s", "s",
                        #                              "o", "o", "o", "o", "o", "o", "o", "o", "o", "o",
                        #                              "p", "p", "p", "p", "p", "p", "p", "p", "p", "p"],
                        #                     s=24,  # s=16 is equal with markersize=4
                        #                     # markers=markers,
                        #                     data=tsne_df)
                        #
                        # plt.xlabel('')
                        # plt.ylabel('')
                        # plt.xticks(fontsize=18)
                        # plt.yticks(fontsize=18)
                        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
                        # # plt.savefig('./tsne_plot/compare_{}.png'.format(model.imp_method), dpi=300, bbox_inches='tight')
                        # plt.savefig('./tsne_plot/task1/compare_{}_runid_{}.pdf'.format(model.imp_method, runid),
                        #             dpi=600, bbox_inches='tight')
                        # plt.close()
                # </editor - fold>

                # <editor-fold desc='tsne-2020-07-18-2  task 17 hou de task 1-9-17'>
                # TSNE code  tsne-2020-07-18-2  task 17 hou de task 1-9-17
                # if True and task == 16:
                #     TSNE_feature = flogits[0].tolist()
                #     print(len(flogits))
                #     print(len(flogits[0]))
                #     ys = np.argmax(fy[0], axis=-1)[:]
                #     print(len(fy))
                #     print(len(fy[0]))
                #     print(ys.shape)
                #     TSNE_label = ys.astype(int).tolist()
                #
                #     TSNE_feature = TSNE_feature + flogits[8].tolist()
                #     ys = np.argmax(fy[8], axis=-1)[:]
                #     ys = ys + (8 + 1) * 100
                #     TSNE_label = TSNE_label + ys.astype(int).tolist()
                #
                #     TSNE_feature = TSNE_feature + flogits[16].tolist()
                #     ys = np.argmax(fy[16], axis=-1)[:]
                #     ys = ys + (16 + 1) * 100
                #     TSNE_label = TSNE_label + ys.astype(int).tolist()
                #
                #
                #     from sklearn.manifold import TSNE
                #     import matplotlib.pyplot as plt
                #     import pandas as pd
                #     import seaborn as sns
                #     # sns.set(font_scale=1.8)
                #     tsne = TSNE(n_components=2, random_state=0)
                #
                #     print('TSNE for task {}-{}'.format(task, 0))
                #     # color1 = ['darkred', 'red', 'firebrick', 'coral', 'lightcoral']
                #     # # 'tomato', 'orangered', '', 'salmon', 'maroon'
                #     # # color1 = ['black', 'dimgray', 'gray', 'darkgray', 'lightgray']
                #     # color2 = ['darkblue', 'blue', 'royalblue', 'steelblue', 'lightsteelblue']
                #     # #  'cyan', 'c', 'darkcyan', 'teal', 'lightblue']
                #     # color3 = ['darkgreen', 'green', 'lime', 'limegreen', 'lightgreen']
                #     # #  'palegreen', '', '', 'greenyellow', ''
                #     # # none sage c g r
                #     # color_a = color1 + color1 + color2 + color2 + color3 + color3
                #     color1 = ['lightcoral', 'indianred', 'brown', 'firebrick', 'maroon',
                #               'red', 'salmon', 'tomato', 'coral', 'orangered']
                #     color2 = ['royalblue', 'cyan', 'navy', 'lightblue', 'deepskyblue', 'skyblue',
                #               'steelblue', 'dodgerblue', 'blue', 'cornflowerblue']  # teal cadetblue
                #     color3 = ['yellowgreen', 'palegreen', 'greenyellow', 'olivedrab',  # 'darkolivegreen'
                #               'lightgreen', 'forestgreen', 'green', 'lime', 'limegreen', 'seagreen']
                #     color_a = color1 + color2 + color3
                #     # color_a = ['red'] * 10 + ['blue'] * 10 + ['green'] * 10
                #     # al = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] * 3
                #
                #     tsne_obj = tsne.fit_transform(TSNE_feature[:])
                #
                #     tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                #                             'Y': tsne_obj[:, 1],
                #                             'digit': np.array(TSNE_label)})
                #     tsne_df.head()
                #     print(list(set(TSNE_label)))
                #     print(len(color_a))
                #     g = sns.scatterplot(x="X", y="Y",
                #                         hue="digit",
                #                         style="digit",
                #                         palette=color_a,
                #                         legend='full',  # `legend` must be 'brief', 'full', or False
                #                         # markers=["o", "o", "o", "o", "o", "x", "x", "x", "x", "x", "1", "1", "1", "1", "1"],
                #                         # alpha=al,
                #                         markers=["o", "o", "o", "o", "o", "o", "o", "o", "o", "o",
                #                                  "o", "o", "o", "o", "o", "o", "o", "o", "o", "o",
                #                                  "o", "o", "o", "o", "o", "o", "o", "o", "o", "o"],
                #                         s=20,  # markersize=4
                #                         data=tsne_df)
                #
                #     plt.xlabel('')
                #     plt.ylabel('')
                #     plt.xticks(fontsize=18)
                #     plt.yticks(fontsize=18)
                #     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
                #     # plt.savefig('./tsne_plot/compare_{}.png'.format(model.imp_method), dpi=300, bbox_inches='tight')
                #     plt.savefig(
                #         './tsne_plot/task1-9-17/compare_{}_runid_{}.png'.format(model.imp_method, runid),
                #         dpi=600, bbox_inches='tight')
                #     plt.close()
                # </editor-fold>

                # <editor-fold desc='new visualization'>
                # if True and task == 16:
                #     net_vals = np.zeros((30000, 2), dtype=np.float32)
                #     net_labels = np.zeros((30000, 1), dtype=np.int32)
                #
                #     net_vals[0:10000, :] = np.array(flogits[0])
                #     ys = np.argmax(fy[0], axis=-1)[:]
                #     net_labels[0:10000, :] = ys.astype(int).reshape(10000, 1)
                #
                #     net_vals[10000:20000, :] = np.array(flogits[8])
                #     ys = np.argmax(fy[8], axis=-1)[:]
                #     net_labels[10000:20000, :] = ys.astype(int).reshape(10000, 1)
                #
                #     net_vals[20000:30000, :] = np.array(flogits[16])
                #     ys = np.argmax(fy[16], axis=-1)[:]
                #     net_labels[20000:30000, :] = ys.astype(int).reshape(10000, 1)
                #
                #     np.save(
                #         '/home/wangshuai/project/RACL/softmaxVis/{}_MNIST_runid_{}_hidden'.format(model.imp_method, runid),
                #         net_vals)
                #     np.save(
                #         '/home/wangshuai/project/RACL/softmaxVis/{}_MNIST_runid_{}_labels'.format(model.imp_method, runid),
                #         net_labels)
                # </editor-fold>

                # <editor-fold desc='theta'>
                # if task == 16:
                #     TSNE_feature = flogits[0].tolist()
                #     print(len(flogits))
                #     print(len(flogits[0]))
                #     print(np.array(flogits).shape)
                #     ys = np.argmax(fy[0], axis=-1)[:]
                #     print(len(fy))
                #     print(len(fy[0]))
                #     print(ys.shape)
                #     TSNE_label = ys.astype(int).tolist()
                #
                #     TSNE_feature = TSNE_feature + flogits[8].tolist()
                #     ys = np.argmax(fy[8], axis=-1)[:]
                #     ys = ys + (8 + 1) * 100
                #     TSNE_label = TSNE_label + ys.astype(int).tolist()
                #
                #     TSNE_feature = TSNE_feature + flogits[16].tolist()
                #     ys = np.argmax(fy[16], axis=-1)[:]
                #     ys = ys + (16 + 1) * 100
                #     TSNE_label = TSNE_label + ys.astype(int).tolist()
                #
                #     # theta_1 = theta[0]
                #     # print(theta_1)
                #     # print(theta_1[0])
                #     # theta_pred = {}
                #     # print('a')
                #     # for i in range(len(theta_1)):
                #     #     idx_t = ys[i]
                #     #     if ys[i] not in theta_pred.keys():
                #     #         theta_pred[ys[i]] = [theta_1[i][idx_t]]
                #     #     else:
                #     #         theta_pred[ys[i]].append(theta_1[i][idx_t])
                #     # f_t = open("theta/theta_1.txt", 'w')
                #     # f_t.write(str(theta_pred))
                #     # f_t.close()
                #
                #     from sklearn.manifold import TSNE
                #     import matplotlib.pyplot as plt
                #     import pandas as pd
                #     import seaborn as sns
                #     # sns.set(font_scale=1.8)
                #     tsne = TSNE(n_components=2, random_state=0)
                #     tsne_obj = tsne.fit_transform(TSNE_feature[:])
                #
                #     tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                #                             'Y': tsne_obj[:, 1],
                #                             'digit': np.array(TSNE_label)})
                #     print(tsne_df.head())
                #     tsne_center = tsne_df.groupby('digit').aggregate({'X': np.mean, 'Y': np.mean})
                #     print(tsne_center)
                #     tsne_center.to_json("theta/theta_1_center_{}_noEMA_seed13.json".format(model.imp_method))
                #     exit()
                # </editor-fold>

                # <editor-fold desc='theta2'>
                # if True or task in [0, 8, 16]:
                #     ys = np.argmax(fy[0], axis=-1)[:]
                #     theta_1 = theta[0]
                #     print(theta_1)
                #     print(theta_1[0])
                #     theta_pred = {}
                #     print('a')
                #     for i in range(len(theta_1)):
                #         idx_t = ys[i]
                #         if ys[i] not in theta_pred.keys():
                #             theta_pred[ys[i]] = [theta_1[i][idx_t]]
                #         else:
                #             theta_pred[ys[i]].append(theta_1[i][idx_t])
                #     f_t = open("theta/theta_m2_{}_task_{}_AKCL.txt".format(args.m2_value, str(int(task)+1)), 'w')
                #     f_t.write(str(theta_pred))
                #     f_t.close()
                #     # ys = np.argmax(fy[8], axis=-1)[:]
                #     # theta_1 = theta[8]
                #     # print(theta_1)
                #     # print(theta_1[0])
                #     # theta_pred = {}
                #     # print('a')
                #     # for i in range(len(theta_1)):
                #     #     idx_t = ys[i]
                #     #     if ys[i] not in theta_pred.keys():
                #     #         theta_pred[ys[i]] = [theta_1[i][idx_t]]
                #     #     else:
                #     #         theta_pred[ys[i]].append(theta_1[i][idx_t])
                #     # f_t = open("theta/theta_9.txt", 'w')
                #     # f_t.write(str(theta_pred))
                #     # f_t.close()
                #     # ys = np.argmax(fy[16], axis=-1)[:]
                #     # theta_1 = theta[16]
                #     # print(theta_1)
                #     # print(theta_1[0])
                #     # theta_pred = {}
                #     # print('a')
                #     # for i in range(len(theta_1)):
                #     #     idx_t = ys[i]
                #     #     if ys[i] not in theta_pred.keys():
                #     #         theta_pred[ys[i]] = [theta_1[i][idx_t]]
                #     #     else:
                #     #         theta_pred[ys[i]].append(theta_1[i][idx_t])
                #     # f_t = open("theta/theta_17.txt", 'w')
                #     # f_t.write(str(theta_pred))
                #     # f_t.close()
                #     # exit()
                # </editor-fold>

            else:
                if MEASURE_PERF_ON_EPS_MEMORY:
                    eps_mem = {
                        'images': episodic_images,
                        'labels': episodic_labels,
                    }
                    # Measure perf on episodic memory
                    ftask = test_task_sequence(model, sess, eps_mem, task, args.online_cross_val)
                else:
                    # List to store accuracy for all the tasks for the current trained model
                    ftask = test_task_sequence(model, sess, datasets, task, args.online_cross_val)

            # Store the accuracies computed at task T in a list
            evals.append(ftask)

            # Reset the optimizer
            model.reset_optimizer(sess)
            # cur_task_grads.append(this_cur_task_grads)
            # cur_task_ids.append(this_cur_task_ids)
            # -> End for loop task
        ########################################
        # gradient TSNE
        if False:
            tsne_obj = tsne.fit_transform(cur_task_grads)

            tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                                    'Y': tsne_obj[:, 1],
                                    'task': cur_task_ids})

            tsne_df.head()
            sns.set_style('darkgrid')

            sns.scatterplot(x="X", y="Y",
                            hue="task",
                            palette="Set2",
                            #   legend='full',
                            data=tsne_df)
            # plt.axis('off')
            plt.savefig('./tsne_plot/{}/class_0.pdf'.format(model.imp_method))
            plt.close()
            exit()

        if False:
            sns.distplot(grads_deviation, shade=True, color='r')

            plt.savefig('./tsne_plot/{}/grads_dev.pdf'.format(model.imp_method))
            plt.close()
            exit()

        ########################################
        runs.append(np.array(evals))
        # End for loop runid

    runs = np.array(runs)
    # print(runs)
    return runs


def test_task_sequence(model, sess, test_data, taskid, cross_validate_mode):
    """
    Snapshot the current performance
    """
    if TIME_MY_METHOD:
        # Only compute the training time
        return np.zeros(model.num_tasks)

    list_acc = []
    list_logit = []
    list_y = []
    list_theta = []
    if model.imp_method == 'PNN':
        pnn_logit_mask = np.ones([model.num_tasks, TOTAL_CLASSES])
    else:
        logit_mask = np.ones(TOTAL_CLASSES)

    if MEASURE_PERF_ON_EPS_MEMORY:
        for task in range(model.num_tasks):
            mem_offset = task * SAMPLES_PER_CLASS * TOTAL_CLASSES
            feed_dict = {model.x: test_data['images'][mem_offset:mem_offset + SAMPLES_PER_CLASS * TOTAL_CLASSES],
                         model.y_: test_data['labels'][mem_offset:mem_offset + SAMPLES_PER_CLASS * TOTAL_CLASSES],
                         model.keep_prob: 1.0,
                         model.output_mask: logit_mask, model.train_phase: False}
            acc = model.accuracy.eval(feed_dict=feed_dict)
            list_acc.append(acc)
        print(list_acc)
        return list_acc

    for task, _ in enumerate(test_data):
        # list_logit = None
        # list_y = None

        if model.imp_method == 'PNN':
            pnn_train_phase = np.array(np.zeros(model.num_tasks), dtype=np.bool)
            feed_dict = {model.x: test_data[task]['test']['images'],
                         model.y_[task]: test_data[task]['test']['labels'], model.keep_prob: 1.0}
            train_phase_dict = {m_t: i_t for (m_t, i_t) in zip(model.train_phase, pnn_train_phase)}
            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, pnn_logit_mask)}
            feed_dict.update(train_phase_dict)
            feed_dict.update(logit_mask_dict)
            acc = model.accuracy[task].eval(feed_dict=feed_dict)
        else:
            feed_dict = {model.x: test_data[task]['test']['images'],
                         model.y_: test_data[task]['test']['labels'], model.keep_prob: 1.0,
                         model.output_mask: logit_mask, model.train_phase: False}
            acc = model.accuracy.eval(feed_dict=feed_dict)
            pruned_logits = model.features.eval(feed_dict=feed_dict)
            # theta = model.thetaw.eval(feed_dict=feed_dict)
            # tmp_logits = model.pruned_logits.eval(feed_dict=feed_dict)
            # print(tmp_logits)
            # print(theta)
            # print(tmp_logits[0])
            # print(theta[0])
            # exit()
            # # pruned_logits = model.net.eval(feed_dict=feed_dict)

        # list_logit = pruned_logits
        # list_y = test_data[task]['test']['labels']
        # # save f to matrix mfr    id-task
        # yp = np.argmax(list_y, axis=-1)[:]
        # yp_dict = dict(collections.Counter(yp))
        # key_list = list(yp_dict.keys())
        # for w, p in enumerate(yp):
        #     MFR_matrix[taskid][task][key_list.index(p)] += list_logit[w]
        # for p in range(5):
        #     MFR_matrix[taskid][task][p] /= yp_dict[key_list[p]]

        list_logit.append(pruned_logits)
        list_y.append(test_data[task]['test']['labels'])
        list_acc.append(acc)
        # list_theta.append(theta)  # theta 10000*10   list_theta 17*10000*10
        # print(theta)
        # print(min(theta))
        # print(max(theta))
        # exit()
        # print(np.array(theta).shape)

    print(np.mean(list_acc), list_acc)
    return list_acc, list_logit, list_y, None


def main():
    """
    Create the model and start the training
    """

    # Get the CL arguments
    args = get_arguments()

    # Check if the network architecture is valid
    if args.arch not in VALID_ARCHS:
        raise ValueError("Network architecture %s is not supported!" % (args.arch))

    # Check if the method to compute importance is valid
    if args.imp_method not in MODELS:
        raise ValueError("Importance measure %s is undefined!" % (args.imp_method))

    # Check if the optimizer is valid
    if args.optim not in VALID_OPTIMS:
        raise ValueError("Optimizer %s is undefined!" % (args.optim))

    # Create log directories to store the results
    if not os.path.exists(args.log_dir):
        print('Log directory %s created!' % (args.log_dir))
        os.makedirs(args.log_dir)

    # Generate the experiment key and store the meta data in a file
    exper_meta_data = {'DATASET': 'PERMUTE_MNIST',
                       'NUM_RUNS': args.num_runs,
                       'TRAIN_SINGLE_EPOCH': args.train_single_epoch,
                       'IMP_METHOD': args.imp_method,
                       'SYNAP_STGTH': args.synap_stgth,
                       'FISHER_EMA_DECAY': args.fisher_ema_decay,
                       'FISHER_UPDATE_AFTER': args.fisher_update_after,
                       'OPTIM': args.optim,
                       'LR': args.learning_rate,
                       'BATCH_SIZE': args.batch_size,
                       'MEM_SIZE': args.mem_size,
                       'M1_VALUE': args.m1_value,
                       'M2_VALUE': args.m2_value,
                       'S_VALUE': args.s_value}
    experiment_id = "PERMUTE_MNIST_HERDING_%s_%s_%s_%s_%s_%r_%s-" % (
        args.arch, args.learning_rate, args.train_single_epoch, args.imp_method,
        str(args.synap_stgth).replace('.', '_'),
        str(args.batch_size), str(args.mem_size)) + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    snapshot_experiment_meta_data(args.log_dir, experiment_id, exper_meta_data)

    # Get the subset of data depending on training or cross-validation mode
    if args.online_cross_val:
        num_tasks = K_FOR_CROSS_VAL
    else:
        num_tasks = NUM_TASKS - K_FOR_CROSS_VAL

    # Variables to store the accuracies and standard deviations of the experiment
    acc_mean = dict()
    acc_std = dict()

    # Reset the default graph
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():

        # Set the random seed
        tf.set_random_seed(args.random_seed)

        # Define Input and Output of the model
        x = tf.placeholder(tf.float32, shape=[None, INPUT_FEATURE_SIZE])
        # x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        if args.imp_method == 'PNN':
            y_ = []
            for i in range(num_tasks):
                y_.append(tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES]))
        else:
            y_ = tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES])

        # Define the optimizer
        if args.optim == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        elif args.optim == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)

        elif args.optim == 'MOMENTUM':
            base_lr = tf.constant(args.learning_rate)
            learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - train_step / training_iters), OPT_POWER))
            opt = tf.train.MomentumOptimizer(args.learning_rate, OPT_MOMENTUM)

        # Create the Model/ contruct the graph
        model = Model(x, y_, num_tasks, opt, args.imp_method, args.synap_stgth, args.fisher_update_after,
                      args.fisher_ema_decay, args.m1_value, args.m2_value, args.s_value, network_arch=args.arch)

        # Set up tf session and initialize variables.
        if USE_GPU:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )

        time_start = time.time()
        with tf.Session(config=config, graph=graph) as sess:
            runs = train_task_sequence(model, sess, args)
            # Close the session
            sess.close()
        time_end = time.time()
        time_spent = time_end - time_start
        print(time_spent)
        print(args.imp_method)
        # with open('time.txt', 'a') as f:
        #     f.write('mnist_m1_{}_m2_{}_s_{}_method_{}time:{}'.format(args.m1_value, args.m2_value,
        #                                                            args.s_value, args.imp_method, time_spent))
        #     f.write('\n')

    # Store all the results in one dictionary to process later
    exper_acc = dict(mean=runs)

    # If cross-validation flag is enabled, store the stuff in a text file
    if args.cross_validate_mode:
        acc_mean = runs.mean(0)
        acc_std = runs.std(0)
        cross_validate_dump_file = args.log_dir + '/' + 'PERMUTE_MNIST_%s_%s' % (args.imp_method, args.optim) + '.txt'
        with open(cross_validate_dump_file, 'a') as f:
            if MULTI_TASK:
                f.write('GPU:{} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {}\n'.format(USE_GPU, args.arch,
                                                                                        args.learning_rate,
                                                                                        args.synap_stgth,
                                                                                        acc_mean[-1, :].mean()))
            else:
                f.write('GPU: {} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {} \t Fgt: {} \t Time: {}\n'.format(USE_GPU,
                                                                                                                args.arch,
                                                                                                                args.learning_rate,
                                                                                                                args.synap_stgth,
                                                                                                                acc_mean[
                                                                                                                -1,
                                                                                                                :].mean(),
                                                                                                                compute_fgt(
                                                                                                                    acc_mean),
                                                                                                                str(
                                                                                                                    time_spent)))

    # Store the experiment output to a file
    snapshot_experiment_eval(args.log_dir, experiment_id, exper_acc)


if __name__ == '__main__':
    main()
