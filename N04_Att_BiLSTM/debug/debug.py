# -*- coding: utf-8 -*-

# @Time    : 2019/8/8
# @Author  : Lattine

# ======================

import os

from tensorflow.python import pywrap_tensorflow

from config import Config

config = Config()

ckpt_path = os.path.join(config.BASE_DIR, config.ckpt_model_path, "text_cnn-500")
print(ckpt_path)
reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor name : ", key)
