'''
author='Alan Liang',
author_email='liangao@sia.cn'
'''

import os
import os.path as osp
import numpy as np

import mmcv

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress

label_map = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
}

def convert_liu_to_coco(data_path):

    modes = ['train', 'val']

    for mode in modes:

        annotations = []
        images = []
        obj_count = 0

        mode_data_path = osp.join(data_path, 'images', mode)
        mode_label_path = osp.join(data_path, 'labels', mode)

        mode_data_list = os.listdir(mode_data_path)
        mode_label_list = os.listdir(mode_label_path)

        assert len(mode_data_list) == len(mode_label_list)

        for idx in range(len(mode_data_list)):
            file_name = osp.join(mode_data_path, mode_data_list[idx])
            label_path = osp.join(mode_label_path, mode_label_list[idx])
            height, width = mmcv.imread(file_name).shape[:2]
            images.append(
                dict(id=idx, file_name=mode_data_list[idx], height=height, width=width))
            
            regions = np.loadtxt(label_path)

            for region in regions:
                x, y, w, h = region[1], region[2], region[3], region[4]

                x1 = (x - w / 2) * width
                y1 = (y - h / 2) * height
                x2 = (x + w / 2) * width
                y2 = (y + h / 2) * height
                box_width = max(0, x2 - x1)
                box_height = max(0, y2 - y1)

                data_anno = dict(
                    image_id=idx,
                    id=obj_count,
                    category_id=label_map[region[0]],
                    bbox=[x1, y1, box_width, box_height],
                    area=box_width * box_height,
                    segmentation=[[x1, y1, x2, y1, x2, y2, x1, y2]],
                    iscrowd=0)
                
                annotations.append(data_anno)
                obj_count += 1

        coco_format_json = dict(
            images=images,
            annotations=annotations,
            categories=[{
                'id': 0,
                'name': 'good'
            },
            {
                'id': 1,
                'name': 'Liu'
            }])
        
        out_file = osp.join('./data/Liu/images/annotations', f'{mode}_coco.json')
        dump(coco_format_json, out_file)

if __name__ == '__main__':
    convert_liu_to_coco('./data/Liu')




            
            











