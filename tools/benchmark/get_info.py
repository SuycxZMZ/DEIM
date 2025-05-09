"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import argparse
from calflops import calculate_flops
from engine.core import YAMLConfig

import torch
import torch.nn as nn
import copy
from typing import Tuple
# from ....engine.misc import stats

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}

def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=None)
    class Model_for_flops(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            outputs = self.model(images)
            return outputs

    args = cfg

    n_parameters, model_stats = stats(cfg)
    print(model_stats)

    # model = Model_for_flops().eval()

    # flops, macs, _ = calculate_flops(model=model,
    #                                  input_shape=(1, 3, 640, 640),
    #                                  output_as_string=True,
    #                                  output_precision=4)
    # params = sum(p.numel() for p in model.parameters())
    # print("Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default= "configs/dfine/dfine_hgnetv2_l_coco.yml", type=str)
    args = parser.parse_args()

    main(args)
