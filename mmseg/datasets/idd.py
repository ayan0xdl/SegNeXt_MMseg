from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class IDDDataset(CustomDataset):
    """IDD-20K dataset (Level 3 labels)."""
    
    CLASSES = (
        'road', 'sidewalk', 'parking', 'rail track', 'person', 'rider', 
        'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle', 
        'caravan', 'trailer', 'traffic light', 'traffic sign', 
        'vegetation', 'terrain', 'sky', 'ground', 'dynamic', 
        'static', 'building', 'wall', 'fence', 'guard rail'
    )
    
    PALETTE = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
        [0, 0, 230], [119, 11, 32], [220, 30, 60], [150, 20, 30], 
        [180, 40, 50], [200, 200, 50], [100, 100, 100], [50, 50, 50], [60, 60, 60]
    ]

    def __init__(self, **kwargs):
        super(IDDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_labels=False,  # IDD starts road at 0
            **kwargs)
