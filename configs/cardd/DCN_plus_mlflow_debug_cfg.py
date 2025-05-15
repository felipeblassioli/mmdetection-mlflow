# Author: felipe.blassioli (modified for debug purposes)
# Time: 2022/11/15 15:43 (original) | Modified: 2025/04/15 17:10
#
# This debug config is based on the original DCN-based Mask R-CNN setup.
# Changes have been made to speed up training:
#   - Reduced max_epochs from 12 to 2.
#   - Checkpoint saving interval increased (every epoch saved).
#   - Log frequency is increased for more immediate feedback.
#   - Learning rate schedule is simplified.
# These modifications let you quickly obtain .pth models for testing inference,
# while keeping the dataset and key model parameters unchanged.
_base_ = [
    "../_base_/datasets/cardd_2017.py",
    "../_base_/default_runtime.py",
    "mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco.py"
]
load_from = f"~/model/pretrained/mask_rcnn_r101_fpn_dconv_c3-c5_1x_coco_20200216-a71f5bce.pth"
resume = False


# Define the damage classes and number of classes.
classes = ("dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat")
n_classes = len(classes)

# Data preprocessor configuration (replaces Normalize and Pad in pipeline)
model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        # Use DCN (Deformable Convolution Networks) with 1 deform group
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(True, True, True, True),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=n_classes,
            # loss calculation is based on the predicted result and the ground truth.
            # The model will be optimized by minimizing the sum of the focal loss, the L1 loss, and the cross-entropy loss.
            # For focal loss, we use the α-balanced version to control the importance of different categories.
            #
            # L_focal = -α(1-p_{i,c})^γ * log(p_{i,c})
            #
            # where p_{i,c} is the predicted probability of the i-th object and the c-th class.
            # α is the balancing factor and γ is the focusing parameter.
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.5,
                loss_weight=1.0,
            ),
        ),
        mask_head=dict(num_classes=n_classes),
    ),
)

# Training and testing configuration
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=2, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# Optimization configuration
optim_wrapper = dict(
    type="OptimWrapper",
    #  Stochastic gradient descent (SGD) is adopted as the optimizer, and we set the weight decay as 0.0001 and the momentum as 0.9.
    optimizer=dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=None,
)

# Parameter scheduler (replaces lr_config)
param_scheduler = [
    dict(type="MultiStepLR", begin=0, end=2, by_epoch=True, milestones=[1], gamma=0.1),
]

# Logging and visualization configuration
vis_backends = [
    dict(type="LocalVisBackend", save_dir="./vis_output"),
    # dict(
    #     type='MLflowVisBackend',
    #     save_dir='temp_dir',
    #     exp_name='DCN-plus',
    #     run_name='mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco',
    #     tags=None,
    #     params=None,
    #     tracking_uri='http://localhost:5000',
    #     artifact_suffix=['.json', '.log', '.py', 'yaml'])
]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")