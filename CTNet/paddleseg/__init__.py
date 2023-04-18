# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import models, datasets, transforms

__version__ = '2.0.0'

#core：包含一些核心的函数和类，例如数据加载、训练和评估模型的函数等。
# cvlibs：包含一些计算机视觉领域的函数和类，例如图像处理、目标检测等。
# datasets：包含一些数据集的类和函数，例如用于图像分类、目标检测、语义分割等领域的数据集，例如COCO、PASCAL VOC、Cityscapes等。
# transforms：包含一些对输入数据进行变换的函数和类，例如对图像进行随机缩放、裁剪、旋转、翻转等，以增加数据多样性和模型鲁棒性。
# utils：包含一些辅助函数和工具类，例如对模型参数进行保存和加载、绘制图表、计算指标等。