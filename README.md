# MMClassification-Tutorial

#### Check nvcc version
!nvcc -V

#### Check GCC version
!gcc --version

#### Check Pytorch installation
import torch, torchvision
print(torch._version)
print(torch.cuda.is_available())

#### Install PyTorch
!pip install -U torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

## 安装MMCV Install mmcv
#### MMCV是OpenMMLab代码库的基础库，Linux环境安装Whl包已经打包好，可以直接下载安装
#### 需要注意PyTorch和CUDA版本，确保能够正常安装
#### 前面的步骤中，输出了CUDA和PyTorch的版本，分别为10.1和1.8.1，需要选择相应的MMCV版本

!pip install mmcv-full

!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html

需要让cu_version和torch_version和平台的CUDA版本和Torch版本保持一致

!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html

mmcv的网址 github.com/open-mmlab/mmcv


## 克隆并安装 mmclssification

#### 下载源码，！表示执行的是python代码以外的命令
!git clone https://github.com/open-mmlab/mmclassification.git

#### 切换到mmclassification目录中 %不是外部命令也不是python命令
%cd mmclassification

#### -e表示可编辑模式
!pip install -e .


#### Check MMClassification installation
import mmcls
print(mmcls.__version__)


## 使用MMCls预训练模型
MMCls提供很多预训练好的模型，这些模型都在ImageNet数据集上获得了state-of-art的结果
可以直接使用这些模型进行推理计算
在使用预训练模型之前，需要进行以下操作：
准备模型
    准备config配置文件
    准备模型权重参数文件
构建模型
进行推理计算


## 准备模型文件
预训练模型通过配置文件和权重参数文件来定义。
配置文件定义了模型结构，权重参数文件保存了训练好的模型参数。
在GitHub上的MMCls通过不同的页面来提供预训练模型。
比如，MobileNetV2的配置文件和权重参数文件在这个链接下https://github.com/open-mmlab/mmclassification/tree/master/configs/mobilenet_v2
在安装mmcls时就已经将配置文件安装到了本地，但是还需要手动下载模型权重参数文件
为方便起见将权重参数文件统一保存到checkpoints文件夹下
!mkdir checkpoints
!wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_batch256_imagenet_20200708-cfb998bf.pth -P checkpoints

确保配置文件和参数文件都存在
!ls configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py
!ls checkpoints/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth


from mmcls.apis import inference_model, init_model, show_result_pyplot

# Specify the path to config file and checkpoint file
config_file = 'configs/resnet/resnet50_b32x8_imagenet.py'
checkpoint_file = 'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
# Specify the device. You may also use cpu by `device='cpu'`.
device = 'cuda:0'
# Build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device=device)


# Test a single image
img = 'demo/demo.JPEG'
result = inference_model(model, img)

# Show the results
show_result_pyplot(model, img, result)

from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

# Pack image info into a dict
data = dict(img_info=dict(filename=img), img_prefix=None)
# Parse the test pipeline
cfg = model.cfg
test_pipeline = Compose(cfg.data.test.pipeline)
# Process the image
data = test_pipeline(data)

# Scatter to specified GPU
data = collate([data], samples_per_gpu=1)
if next(model.parameters()).is_cuda:
    data = scatter(data, [device])[0]
    
    # Forward the model
with torch.no_grad():
    features = model.extract_feat(data['img'])

# Show the feature, it is a 1280-dim vector
print(features.shape)

!wget https://www.dropbox.com/s/ckv2398yoy4oiqy/cats_dogs_dataset.zip?dl=0 -O cats_dogs_dataset.zip
!mkdir data
!unzip -q cats_dogs_dataset.zip -d ./data/cats_dogs_dataset/

# Let's take a look at the dataset
import mmcv
import matplotlib.pyplot as plt

img = mmcv.imread('data/cats_dogs_dataset/training_set/training_set/cats/cat.1.jpg')
plt.figure(figsize=(8, 6))
plt.imshow(mmcv.bgr2rgb(img))
plt.show()

import shutil
import os
import os.path as osp


data_root = './data/cats_dogs_dataset/'
train_dir = osp.join(data_root, 'training_set/training_set/')
val_dir = osp.join(data_root, 'val_set/val_set/')

# Split train/val set
mmcv.mkdir_or_exist(val_dir)
class_dirs = [
    d for d in os.listdir(train_dir) if osp.isdir(osp.join(train_dir, d))
]
for cls_dir in class_dirs:
    train_imgs = [filename for filename in mmcv.scandir(osp.join(train_dir, cls_dir), suffix='.jpg')]
    # Select first 4/5 as train set and the last 1/5 as val set
    train_length = int(len(train_imgs)*4/5)
    val_imgs = train_imgs[train_length:]
    # Move the val set into a new dir
    src_dir = osp.join(train_dir, cls_dir)
    tar_dir = osp.join(val_dir, cls_dir)
    mmcv.mkdir_or_exist(tar_dir)
    for val_img in val_imgs:
        shutil.move(osp.join(src_dir, val_img), osp.join(tar_dir, val_img))
        
        import shutil
import os
import os.path as osp

from itertools import chain


# Generate mapping from class_name to label
def find_folders(root_dir):
    folders = [
        d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


# Generate annotations
def gen_annotations(root_dir):
    annotations = dict()
    folder_to_idx = find_folders(root_dir)
    
    for cls_dir, label in folder_to_idx.items():
        cls_to_label = [
            '{} {}'.format(osp.join(cls_dir, filename), label) 
            for filename in mmcv.scandir(osp.join(root_dir, cls_dir), suffix='.jpg')
        ]
        annotations[cls_dir] = cls_to_label
    return annotations


data_root = './data/cats_dogs_dataset/'
val_dir = osp.join(data_root, 'val_set/val_set/')
test_dir = osp.join(data_root, 'test_set/test_set/')
    
# Save val annotations
with open(osp.join(data_root, 'val.txt'), 'w') as f:
    annotations = gen_annotations(val_dir)
    contents = chain(*annotations.values())
    f.writelines('\n'.join(contents))
    
# Save test annotations
with open(osp.join(data_root, 'test.txt'), 'w') as f:
    annotations = gen_annotations(test_dir)
    contents = chain(*annotations.values())
    f.writelines('\n'.join(contents))

# Generate classes
folder_to_idx = find_folders(train_dir)
classes = list(folder_to_idx.keys())
with open(osp.join(data_root, 'classes.txt'), 'w') as f:
    f.writelines('\n'.join(classes))
    
    # Generate annotations
import os
import mmcv
import os.path as osp

from itertools import chain


# Generate mapping from class_name to label
def find_folders(root_dir):
    folders = [
        d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


# Generate annotations
def gen_annotations(root_dir):
    annotations = dict()
    folder_to_idx = find_folders(root_dir)
    
    for cls_dir, label in folder_to_idx.items():
        cls_to_label = [
            '{} {}'.format(osp.join(cls_dir, filename), label) 
            for filename in mmcv.scandir(osp.join(root_dir, cls_dir), suffix='.jpg')
        ]
        annotations[cls_dir] = cls_to_label
    return annotations


data_root = './data/cats_dogs_dataset/'
train_dir = osp.join(data_root, 'training_set/training_set/')
test_dir = osp.join(data_root, 'test_set/test_set/')

# Generate class list
folder_to_idx = find_folders(train_dir)
classes = list(folder_to_idx.keys())
with open(osp.join(data_root, 'classes.txt'), 'w') as f:
    f.writelines('\n'.join(classes))
    
# Generate train/val set randomly
annotations = gen_annotations(train_dir)
# Select first 4/5 as train set
train_length = lambda x: int(len(x)*4/5)
train_annotations = map(lambda x:x[:train_length(x)], annotations.values())
val_annotations = map(lambda x:x[train_length(x):], annotations.values())
# Save train/val annotations
with open(osp.join(data_root, 'train.txt'), 'w') as f:
    contents = chain(*train_annotations)
    f.writelines('\n'.join(contents))
with open(osp.join(data_root, 'val.txt'), 'w') as f:
    contents = chain(*val_annotations)
    f.writelines('\n'.join(contents))
    
# Save test annotations
test_annotations = gen_annotations(test_dir)
with open(osp.join(data_root, 'test.txt'), 'w') as f:
    contents = chain(*test_annotations.values())
    f.writelines('\n'.join(contents))
    
    import mmcv
import numpy as np

from mmcls.datasets import DATASETS, BaseDataset


# Regist model so that we can access the class through str in configs
@DATASETS.register_module()
class CatsDogsDataset(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            # The ann_file is the annotation files we generate above.
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos
            
            # Load the existing config file
from mmcv import Config
cfg = Config.fromfile('configs/resnet/resnet50_b32x8_imagenet.py')
