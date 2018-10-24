#!/usr/bin/env python

import os
from decimal import Decimal

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Option:
    def __init__(self):
        self.tries = int(5e2)
        self.learning_rate = 5e-3
        self.x_min = -100
        self.x_max = 100
        self.max_noise = 0.1
        self.normalized_x_min = -1
        self.normalized_x_max = 1
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.batch_norm = False
        self.train_size = 49000
        self.test_size = 1000
        self.reg = 1e-1
        self.layer_info = None
        self.f = None
        self.print_in_line = "PYCHARM_HOSTED" not in os.environ
        self.print_frequency = 100
        self.tuning_params = None
        self.show_history_plot = True

    @property
    def tuning_id(self):
        if not self.tuning_params:
            return ''
        tuning_id = ''
        for name, value in self.tuning_params.items():
            tuning_id += f'{name}: {Decimal(value):.2E} '

        return tuning_id


plt_row_size = 2.7
plt_col_size = 2.5
min_plt_width = plt_col_size * 2


def create_options(mode, probe=False):
    options = []

    if mode == 0:
        option = Option()
        option.layer_info = [(10, tf.nn.relu)]
        option.f = lambda x: 4 * x
        options.append(option)

        option = Option()
        option.layer_info = [(10, tf.nn.relu)]
        option.f = lambda x: 4 * x
        option.batch_norm = True
        options.append(option)
    if mode == 1:
        option = Option()
        option.tries = int(1e4)
        option.layer_info = [(1000, tf.nn.relu)]
        option.f = lambda x: pow(x, 2)
        options.append(option)
    elif mode == 2:
        option = Option()
        option.tries = int(5e3)
        option.learning_rate = 1e-2
        option.layer_info = [(1000, tf.nn.relu), (1000, tf.nn.relu)]
        option.f = lambda x: pow(x, 3)
        options.append(option)
    elif mode == 3:
        sample_count = 100 if probe else 1
        tries = 200 if probe else int(1e4)

        for i in range(sample_count):
            if probe:
                lr = 10**np.random.uniform(-3, 0)
                reg = 10**np.random.uniform(-5, 5)
            else:
                lr = 2.09E-2
                reg = 1e-2

            option = Option()
            option.x_min = -200
            option.x_max = 200
            option.tries = int(tries)
            option.normalized_x_max = 100
            option.normalized_x_min = -100
            option.max_noise = 1
            option.layer_info = [(400, tf.nn.relu), (200, tf.nn.relu), (100, tf.nn.relu)]
            option.learning_rate = lr
            option.reg = reg
            option.f = lambda x: x * (x - 40) * (x + 50)
            if probe:
                option.tuning_params = {'lr': lr, 'reg': reg}
                option.show_history_plot = False
            options.append(option)

    return options


def create_data_set(data_set_size, option, max_noise=None):
    noises = create_noises(data_set_size, max_noise)

    xs = np.random.uniform(option.x_min, high=option.x_max, size=data_set_size).reshape((-1, 1)).astype(np.float64)
    ys = option.f(xs) + noises

    return xs, ys


def create_dummy_data_set(data_set_size, option, max_noise=None):
    noises = create_noises(data_set_size, max_noise)

    xs = np.random.randint(option.x_min, high=option.x_max, size=data_set_size).reshape((-1, 1)).astype(np.float64)
    ys = option.f(xs) + noises

    return xs, ys


def create_noises(data_set_size, max_noise=None):
    if max_noise:
        noises = max_noise * 2 * (np.random.rand(data_set_size, 1) - 0.5)
    else:
        noises = np.zeros((data_set_size, 1), dtype=np.float64)

    return noises


def create_placeholders(option):
    xs, ys = create_dummy_data_set(2, option)

    x_shape = list(xs.shape)
    y_shape = list(ys.shape)
    x_shape[0] = None
    y_shape[0] = None
    x_placeholder = tf.placeholder(tf.float64, x_shape)
    y_placeholder = tf.placeholder(tf.float64, y_shape)

    return x_placeholder, y_placeholder


def normalize(t, high, low, normalized_max, normalized_min):
    t = (t - low) / (high - low)
    t = t * (normalized_max - normalized_min) + normalized_min

    return t


def denormalize(t, high, low, normalized_max, normalized_min):
    t = (t - normalized_min) / (normalized_max - normalized_min)
    t = t * (high - low) + low

    return t


def normalize_x(t, option):
    return normalize(t, option.x_max, option.x_min, option.normalized_x_max, option.normalized_x_min)


def normalize_y(t, option):
    low, high = option.f(option.x_min), option.f(option.x_max)
    return normalize(t, high, low, option.normalized_x_max, option.normalized_x_min)


def denormalize_x(t, option):
    return denormalize(t, option.x_max, option.x_min, option.normalized_x_max, option.normalized_x_min)


def denormalize_y(t, option):
    low, high = option.f(option.x_min), option.f(option.x_max)
    return denormalize(t, high, low, option.normalized_x_max, option.normalized_x_min)


def build_network(x_placeholder, y_placeholder, option, run_name):
    normalized_x, normalized_y = normalize_x(x_placeholder, option), normalize_y(y_placeholder, option)

    out = normalized_x

    for unit, activation in option.layer_info:
        _, out = dense(unit, activation, option.initializer, out, option.reg, run_name)
        if option.batch_norm:
            out = tf.layers.batch_normalization(out, training=True)
    _, out = dense(1, None, option.initializer, out, option.reg, run_name)

    loss = tf.losses.mean_squared_error(normalized_y, out)

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=option.learning_rate).minimize(loss, global_step=global_step)

    return out, optimizer, loss, global_step


def predict_graph(sess, x_placeholder, out, option):
    xs = np.arange(option.x_min, option.x_max).reshape((-1, 1))
    ys = sess.run(out, feed_dict={x_placeholder: xs})

    ys = denormalize_y(ys, option)

    return xs, ys


def dense(unit, activation, initializer, input_tensor, reg, run_name):
    # TODO: Distinguish regularizer by each try
    regularizer = tf.contrib.layers.l2_regularizer(reg, scope=run_name)
    layer = tf.layers.Dense(unit, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer,
                            bias_initializer=initializer)
    out = layer.apply(input_tensor)

    return layer, out


def sample_random_batch(xs, ys, batch_size=100):
    population_size = len(xs)
    mask = np.random.choice(population_size, batch_size)
    return xs[mask], ys[mask]


def train(sess, option, x_placeholder, y_placeholder, optimizer, loss, global_step, run_name):
    # train_set = create_dummy_data_set(option.train_size, option, max_noise=option.max_noise)
    # test_set = create_dummy_data_set(option.test_size, option, max_noise=option.max_noise)
    train_set = create_data_set(option.train_size, option, max_noise=option.max_noise)
    test_set = create_data_set(option.test_size, option, max_noise=option.max_noise)

    xs_train, ys_train = train_set
    xs_test, ys_test = test_set

    train_losses = []
    test_losses = []
    best_train_loss = best_test_loss = np.iinfo(np.int32).max

    for i in range(option.tries):
        x_train_batch, y_train_batch = sample_random_batch(xs_train, ys_train)
        x_test_batch, y_test_batch = sample_random_batch(xs_test, ys_test)

        # TODO: Loss goes down drastically
        train_loss, _, steps = sess.run([loss, optimizer, global_step],
                                        feed_dict={
                                            x_placeholder: x_train_batch,
                                            y_placeholder: y_train_batch})
        test_loss = sess.run(loss,
                             feed_dict={
                                 x_placeholder: x_test_batch,
                                 y_placeholder: y_test_batch})

        for reg_loss_tensor in tf.losses.get_regularization_losses(scope=run_name):
            reg_loss = sess.run(reg_loss_tensor)
            train_loss += reg_loss
            test_loss += reg_loss

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        best_train_loss = min(best_train_loss, train_loss)
        best_test_loss = min(best_test_loss, test_loss)

        msg = 'progress: {p:6.2f}%  train loss: {t_l:8.3f}  test loss: {v_l:8.3f}'.format(
            p=100 * (i + 1) / option.tries, t_l=train_loss, v_l=test_loss)
        if option.tuning_params:
            continue
        if option.print_in_line:
            print(msg, end='\n' if i == option.tries - 1 else '\r')
        else:
            if option.print_frequency > 0 and i % option.print_frequency == 0:
                print(msg)

    return train_losses, test_losses, best_train_loss, best_test_loss


def show_histories(histories, options):
    labels = [option.tuning_id for option in options if option.show_history_plot]
    histories = [histories[i] for i in range(len(options)) if options[i].show_history_plot]
    history_cnt = len(histories)
    batch_size = 5

    figure_idx = 0

    for batch_start_idx in range(0, history_cnt, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, history_cnt)
        history_batch = histories[batch_start_idx:batch_end_idx]
        label_batch = labels[batch_start_idx:batch_end_idx]

        show_history_batch(history_batch, label_batch, figure_idx)

        figure_idx += 1


def show_history_batch(batch, labels, figure_idx):
    history_cnt = len(batch)
    row_size = 3

    fig = plt.figure(figure_idx, figsize=(max(plt_col_size * history_cnt, min_plt_width), plt_row_size * row_size))
    for history_idx in range(history_cnt):
        label = labels[history_idx]
        col = history_idx
        xs, ys, train_losses, test_losses = batch[history_idx]
        for row in range(row_size):
            graph = plt.subplot2grid((row_size, history_cnt), (row, col))

            if row == 0:
                graph.set_title('train loss')
                graph.plot(train_losses)
            elif row == 1:
                graph.set_title('test loss')
                graph.plot(test_losses)
            else:
                graph.set_title('prediction')
                graph.annotate(label, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
                graph.scatter(xs, ys)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()


def hide_tick_labels(fig):
    for i, ax in enumerate(fig.axes):
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


def print_losses(option, all_losses, progress):
    train_losses, test_losses, best_train_loss, best_test_loss = all_losses
    if option.tuning_params:
        prefix = ""
        for name, value in option.tuning_params.items():
            prefix += f"{name}: {Decimal(value):.2E} "
        msg = "{}best train loss: {b_tr_l:8.3f}  best test loss: {b_te_l:8.3f}\t({tries}/{total})" \
            .format(prefix, b_tr_l=best_train_loss, b_te_l=best_test_loss, tries=progress[0] + 1, total=progress[1])
        print(msg)
    else:
        msg = "train loss: {t_l:8.3f}  test loss: {v_l:8.3f}" \
            .format(t_l=train_losses[-1], v_l=test_losses[-1])
        print(msg)


def print_best_val_losses(histories, options, size=20):
    best_val_losses = [sorted(history[-1], reverse=True)[-1] for history in histories]
    tuning_ids = [option.tuning_id for option in options]
    sorted_losses_with_ids = sorted(zip(best_val_losses, tuning_ids), key=lambda v: v[0])
    best_losses = sorted_losses_with_ids[:size]

    for loss, tuning_id in best_losses:
        print('{}: {}'.format(tuning_id, loss))


def main():
    options = create_options(3, probe=False)
    histories = []
    for i, option in enumerate(options):
        with tf.Session() as sess:
            run_name = str(i)

            x_placeholder, y_placeholder = create_placeholders(option)
            out, optimizer, loss, global_step = build_network(x_placeholder, y_placeholder, option, run_name)

            sess.run(tf.global_variables_initializer())

            all_losses = train(sess, option, x_placeholder, y_placeholder, optimizer, loss, global_step, run_name)

            train_losses, test_losses, _, _ = all_losses
            print_losses(option, all_losses, (i, len(options)))

            xs, ys = predict_graph(sess, x_placeholder, out, option)
            histories.append((xs, ys, train_losses, test_losses))

    show_histories(histories, options)
    options = [option for option in options if option.tuning_params is not None]
    print_best_val_losses(histories, options)


if __name__ == '__main__':
    main()
