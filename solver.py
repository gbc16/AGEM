"""
AGEM Algorithm

Copyright (C) 2019, Bichuan Guo <gbc16@mails.tsinghua.edu.cn>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from numpy.random import RandomState
from model import reconstruction_error
import time


def solve_agem(net, x_true, sigma_noise,
               sigma_train,
               sigma_noise_hat_init,
               sigma_proposal=0.01,
               type_proposal='mala',
               candidate='mean',
               em_epochs=20, sample_epochs=100):

    variance_train = sigma_train ** 2
    variance_proposal = sigma_proposal ** 2

    variance_noise_hat = np.ones(tuple([x_true.shape[0]] + [1] * (x_true.ndim - 1))) * (sigma_noise_hat_init ** 2)

    rng = RandomState(1)
    H_x_true = x_true
    noise = rng.randn(*H_x_true.shape) * sigma_noise
    y = H_x_true + noise

    x = np.zeros_like(x_true)
    sum_axis = tuple(range(1, x.ndim))

    for em_epoch in range(em_epochs):
        time_start = time.time()

        # posterior sampling
        sample_epochs_this = sample_epochs
        sample_start = sample_epochs_this // 5
        samples = np.zeros((sample_epochs_this, *x.shape))
        forward_samples = np.zeros((sample_epochs_this, *y.shape))

        for sample_epoch in range(sample_epochs_this):
            rec_error = reconstruction_error(net, x)

            x_prop = x
            # random walk MH
            if type_proposal == 'random walk':
                x_prop = x + np.random.randn(*x.shape) * sigma_proposal
                forward_x = x
                forward_samples[sample_epoch, ...] = forward_x
                forward_x_prop = x_prop

            elif type_proposal == 'mala':
                forward_x = x
                forward_samples[sample_epoch, ...] = forward_x
                error_whitened = (y - forward_x) / variance_noise_hat
                grad_log_posterior_x = error_whitened + rec_error / variance_train
                x_prop = x + 0.5 * variance_proposal * grad_log_posterior_x
                x_prop += np.random.randn(*x.shape) * sigma_proposal

                forward_x_prop = x_prop
                error_whitened_prop = (y - forward_x_prop) / variance_noise_hat
                rec_error_prop = rec_error
                grad_log_posterior_x_prop = error_whitened_prop + rec_error_prop / variance_train  # approx.

            log_alpha = np.zeros(x.shape[0])

            ll_x = 0.5 * (((y - forward_x) ** 2) / variance_noise_hat).sum(axis=sum_axis)
            ll_x_prop = 0.5 * (((y - forward_x_prop) ** 2) / variance_noise_hat).sum(axis=sum_axis)
            log_alpha += ll_x - ll_x_prop

            # prior
            lprior = (rec_error * (x_prop - x)).sum(axis=sum_axis) / variance_train
            log_alpha += lprior

            # proposal
            if type_proposal == 'mala':
                lp_x_prop = (0.5 * ((x_prop - x - 0.5 * variance_proposal * grad_log_posterior_x) ** 2) / variance_proposal).sum(axis=sum_axis)
                lp_x = (0.5 * ((x - x_prop - 0.5 * variance_proposal * grad_log_posterior_x_prop) ** 2) / variance_proposal).sum(axis=sum_axis)
                log_alpha += lp_x_prop - lp_x

            replace = np.random.uniform(0, 1, x.shape[0]) < np.exp(log_alpha)
            x[replace] = x_prop[replace]
            samples[sample_epoch, ...] = x

        # MLE
        samples = samples[sample_start::, ...]
        forward_samples = forward_samples[sample_start::, ...]
        error = y - forward_samples
        variance_noise_hat = np.mean((error ** 2), axis=tuple([0] + list(range(2, x.ndim + 1))), keepdims=True)[0]

        # compute metric
        if candidate == 'mean':
            mse = ((samples.mean(0) - x_true) ** 2).mean(sum_axis)
        elif candidate == 'median':
            mse = ((np.median(samples, axis=0) - x_true) ** 2).mean(sum_axis)
        elif candidate == 'last':
            mse = ((samples[-1] - x_true) ** 2).mean(sum_axis)
        elif candidate == 'first':
            mse = ((samples[0] - x_true) ** 2).mean(sum_axis)
        else:
            raise NotImplementedError

        time_end = time.time()
        print('epoch %d, time %.1f sec | ' % (em_epoch, time_end - time_start), end='')
        print('noise_gt: %.4f, noise_est: %.4f (%.4f) | ' %
              (sigma_noise[0], np.sqrt(variance_noise_hat).mean(), np.sqrt(variance_noise_hat).std()), end='')
        print('rmse: %.4f (%.4f)' %
              (np.sqrt(mse).mean(), np.sqrt(mse).std()))

    return np.sqrt(mse).mean(), np.sqrt(mse).std(), np.sqrt(variance_noise_hat).mean(), np.sqrt(variance_noise_hat).std(), variance_noise_hat
