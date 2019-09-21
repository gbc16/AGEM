"""
Test script for reproducing Table 1, AGEM: Solving Linear Inverse Problems via Deep Priors and Sampling, NeurIPS, 2019.

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

from model import AutoEncoder, load_pretrained_model
from solver import solve_agem
import numpy as np
import pickle


def run_test():

    model = AutoEncoder(n=2, dim_x=50, dim_hidden=2000)
    device_id = 0
    model_file = 'model/dae.pth'
    net = load_pretrained_model(model, model_file, device_id)

    x_test = pickle.load(open('data/manifold_test.pkl', 'rb'))
    x_true = x_test[:]

    noise_shape = x_true.shape[1:]
    n_dim = np.prod(noise_shape)
    sigma_train = 0.01
    sigma_proposal = 0.01
    sigma_noise_hat_init = 0.01

    for noise in [0.01, 0.02, 0.03, 0.04]:

        sigma_noise = [noise] * n_dim
        sigma_noise = np.array(sigma_noise[:n_dim]).reshape(noise_shape)

        rmse_mean, rmse_std, noise_mean, noise_std, variance_noise_hat_em = \
            solve_agem(net=net, x_true=x_true, sigma_noise=sigma_noise,
                       sigma_train=sigma_train, sigma_noise_hat_init=sigma_noise_hat_init,
                       sigma_proposal=sigma_proposal, type_proposal='mala',
                       candidate='mean', em_epochs=10, sample_epochs=1000)

        print('[AGEM] noise_gt: %.2f | rmse %.4f (%.4f), noise_est: %.4f (%.4f)' % (
            noise, rmse_mean, rmse_std, noise_mean, noise_std
        ))


if __name__ == '__main__':
    run_test()
