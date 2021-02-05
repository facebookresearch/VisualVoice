#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
	def __init__(self):
		super(BaseDataset, self).__init__()

	def name(self):
		return 'BaseDataset'

	def initialize(self, opt):
		pass