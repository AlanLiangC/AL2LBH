from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset

@DATASETS.register_module()
class LiuDataset(CocoDataset):
    """Dataset for Liu."""

    METAINFO = {
        'classes': ('good', 'Liu'),
        'palette':
        [(220, 20, 60), (119, 11, 32)]
    }