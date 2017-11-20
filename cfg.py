import os

home = os.path.expanduser('~')

crop = 1024

pre_resize_w = 1280
pre_resize_h = 640

test_w = 1280
test_h = 640

size = 512

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

n_class = 34  # including background

coco_root = os.path.join(home, 'data', 'COCO')
coco_classes = ['car', 'bus', 'truck']
coco_cat_ids = [3, 6, 8]
coco_map = dict(zip(coco_cat_ids, range(1, len(coco_cat_ids) + 1)))

voc_root = os.path.join(home, 'data', 'VOCdevkit')
ade_root = os.path.join(home, 'data', 'scene_parsing')

cityscapes_root = os.path.join(home, 'data', 'cityscapes')
