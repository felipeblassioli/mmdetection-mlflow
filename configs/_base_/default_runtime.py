default_scope = 'mmdet' # not sure this is needed

default_hooks = dict(
    # `IterTimerHook`, `ParamSchedulerHook` and `DistSamplerSeedHook` are simple and no need to be modified.
    timer=dict(type="IterTimerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # LoggerHook: enables to set intervals
    # Detail usages can be found in the [docstring](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L18).
    logger=dict(type="LoggerHook", interval=50),
    # CheckpointHook: saving checkpoints periodically
    # [`CheckpointHook`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19) provides other options such as `max_keep_ckpts`, `save_optimizer` and etc. The users could set `max_keep_ckpts` to only save small number of checkpoints or decide whether to store state dict of optimizer by `save_optimizer`. More details of the arguments are [here](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/checkpoint_hook.py#L19)
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="auto",
        save_optimizer=True,
    ),
    # DetVisualizationHook: visualize validation and testing process prediction results
    # """Detection Visualization Hook. Used to visualize validation and testing
    # process prediction results.
    #
    # In the testing phase:
    # 1. If ``show`` is True, it means that only the prediction results are
    #     visualized without storing data, so ``vis_backends`` needs to
    #     be excluded.
    # 2. If ``test_out_dir`` is specified, it means that the prediction results
    #     need to be saved to ``test_out_dir``. In order to avoid vis_backends
    #     also storing data, so ``vis_backends`` needs to be excluded.
    # 3. ``vis_backends`` takes effect if the user does not specify ``show``
    #     and `test_out_dir`. You can set ``vis_backends`` to WandbVisBackend or
    #     TensorboardVisBackend to store the prediction result in Wandb or
    #     Tensorboard.
    #
    # Args:
    #     draw (bool): whether to draw prediction results. If it is False,
    #         it means that no drawing will be done. Defaults to False.
    #     interval (int): The interval of visualization. Defaults to 50.
    #     score_thr (float): The threshold to visualize the bboxes
    #         and masks. Defaults to 0.3.
    #     show (bool): Whether to display the drawn image. Default to False.
    #     wait_time (float): The interval of show (s). Defaults to 0.
    #     test_out_dir (str, optional): directory where painted images
    #         will be saved in testing process.
    #     backend_args (dict, optional): Arguments to instantiate the
    #         corresponding backend. Defaults to None.
    # """
    visualization=dict(type="DetVisualizationHook"),
)

# Runtime configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# Log processor configuration
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False