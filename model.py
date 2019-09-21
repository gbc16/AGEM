"""
Definition of denoising autoencoders

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

import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, n, dim_x, dim_hidden):
        super().__init__()
        encoder_layers = [nn.Linear(dim_x, dim_hidden)]
        for _ in range(n - 1):
            encoder_layers += [nn.ReLU(),
                               nn.Linear(dim_hidden, dim_hidden)]
        encoder_layers += [nn.Tanh()]
        decoder_layers = []
        for _ in range(n - 1):
            decoder_layers += [nn.Linear(dim_hidden, dim_hidden),
                               nn.ReLU()]
        decoder_layers += [nn.Linear(dim_hidden, dim_x)]
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        h = self.encoder(x)
        x_rec = self.decoder(h)

        return x_rec


def load_pretrained_model(_model, model_file, _device_id=0):
    _device = torch.device("cuda:%d" % _device_id if torch.cuda.is_available() else "cpu")

    _net = nn.DataParallel(_model, device_ids=[_device_id])
    _net.load_state_dict(torch.load(model_file, map_location='cpu'))
    _net.to(_device)

    return _net


def reconstruction_error(_net, _x):
    _net.eval()
    with torch.no_grad():
        r_x = _net(torch.from_numpy(_x).float()).detach().cpu().numpy()
    return r_x - _x
