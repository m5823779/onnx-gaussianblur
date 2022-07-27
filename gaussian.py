# make onnx ppp models (ppp - pre/post process)

import onnx
from onnxconverter_common import float16

import os
import torch
import argparse
import numpy as np
from torchvision import transforms


class PPPBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bm_input_h = 256
        self.bm_input_w = 256

        self.bm_input_names = None
        self.bm_input_axes = None
        self.bm_output_names = None


class GaussianBlur(PPPBaseModel):
    def __init__(self, args):
        super().__init__()

        kx, ky = args.filter, args.filter

        input_name = f"GaussianBlur_{kx}x{ky}"


        self.m_input_names = [input_name]
        self.m_output_names = ['output']

        if (args.input_n):
            dynamic_axes = {1: 'input_h', 2: 'input_w'} if args.channel_last else {2: 'input_h', 3: 'input_w'}
        else:
            dynamic_axes = {0: 'input_h', 1: 'input_w'} if args.channel_last else {1: 'input_h', 2: 'input_w'}
        self.m_dynamic_axes = {input_name: dynamic_axes, 'output': dynamic_axes}

        self.gauss_t = transforms.GaussianBlur(kernel_size=(kx, ky), sigma=(0.001, 0.001))

    def forward(self, x):
        x = self.gauss_t(x)
        return x



def export_model(args):
    model = GaussianBlur(args)  # create obj instance

    net_w = 256 if args.input_w == 0 else args.input_w
    net_h = 256 if args.input_h == 0 else args.input_h

    num_channels = args.input_c

    input_shape = [net_h, net_w, num_channels] if args.channel_last else [num_channels, net_h, net_w]

    # Create dummy input
    img_input = np.zeros(input_shape, np.float32)

    # Add batch channel (N*C*H*W, N*H*W*C)
    if args.input_n:
        img_input = np.expand_dims(img_input, axis=0)

    # Transform numpy array to torch tensor
    sample = torch.from_numpy(img_input)

    save_path = f"GaussianBlur({args.filter}x{args.filter})-{args.input_w}x{args.input_h}.onnx"
    save_path_fp16 = f"GaussianBlur({args.filter}x{args.filter})-{args.input_w}x{args.input_h}_fp16.onnx"

    if args.output_path != None:
        save_path = args.output_path + "/" + save_path
        save_path_fp16 = args.output_path + "/" + save_path_fp16

    torch.onnx.export(model, sample, save_path,
                      input_names=model.m_input_names,
                      output_names=model.m_output_names,
                      dynamic_axes=model.m_dynamic_axes,
                      opset_version=args.opset_version,
                      do_constant_folding=False,
                      )

    print(f"saved {save_path}")

    # float16
    model_fp32 = onnx.load(save_path)
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, save_path_fp16)
    print(f"saved {save_path_fp16}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', default='onnx_models', help='folder for exporting onnx model file')
    parser.add_argument('-v', '--opset_version', default=10, type=int, help='onnx opset version')

    parser.add_argument('-width', '--input_w', default=0, type=int, help='input node shape width')
    parser.add_argument('-height', '--input_h', default=0, type=int, help='input node shape height')
    parser.add_argument('-c', '--input_c', default=1, type=int, help='input node shape channel')
    parser.add_argument('-n', '--input_n', action='store_true', default=False, help='add batch channel')

    parser.add_argument('-l', '--channel_last', action='store_true', default=False, help='input data is channel last')
    parser.add_argument('-f', '--filter', default=11, type=int, help='gaussian blur filter size')
    args = parser.parse_args()

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    export_model(args)








