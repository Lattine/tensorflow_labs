# -*- coding: utf-8 -*-

# @Time    : 2019/7/30
# @Author  : Lattine

# ======================
import os
from .base import TestDataBase


class TestData(TestDataBase):
    def __init__(self, config):
        super(TestData, self).__init__(config)

        self._test_data_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), config.test_data)
        self._output_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd(), config.output_path)))
        self._sequence_length = config.sequence_length
