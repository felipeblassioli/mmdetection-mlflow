# Using public dataset: https://cardd-ustc.github.io/
DATA_ROOT = "data/CarDD_COCO/"

dataset_type = 'CocoDataset'
classes = ("dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat")
n_classes = len(classes)

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    # DCN+.
    # In our dataset, the object scales of crack, scratch, and dent classes are diverse.
    # This characteristic leads the model can hardly recognize the objects of these hard classes.
    #
    # DCN+ involves two key techniques:
    # 1. multi-scale learning [35] that can be used to handle the scale diversity of objects
    # 2. focal loss [36] to enforce the model to focus on hard categories.
    #
    # Given the image, we first apply multi-scale resizing as the augmentation approach,
    # leading to a more diverse input image size distribution without introducing
    # additional computational or time costs.
    #
    # To be specific, for multi-scale learning, we randomly resize the height of each
    # training image in the range of [640, 1200] while keeping the width as 1333.
    # Then the input will be fed into the backbone model which keeps the same as in DCN [33].
    # The network generates the predicted result, including the class, the bounding box location,
    # and the mask of an object.
    dict(type="RandomResize", scale=[(800, 1333), (1024, 1333)], keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

# Test pipeline
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1024, 1333), keep_ratio=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# Dataloader configuration
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    #sampler=dict(type="DefaultSampler", shuffle=True),
    sampler=dict(type="SubsetSampler", shuffle=True, subset_ratio=0.1, seed=42),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    #sampler=dict(type="DefaultSampler", shuffle=False),
    sampler=dict(type="SubsetSampler", shuffle=True, subset_ratio=0.1, seed=42),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    # sampler=dict(type="DefaultSampler", shuffle=False),
    sampler=dict(type="SubsetSampler", shuffle=True, subset_ratio=0.1, seed=42),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        ann_file="annotations/instances_test2017.json",
        data_prefix=dict(img="test2017/"),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

# Evaluator configuration
val_evaluator = dict(
    type="CocoMetric",
    ann_file=DATA_ROOT + "annotations/instances_val2017.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=DATA_ROOT + "annotations/instances_test2017.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)

