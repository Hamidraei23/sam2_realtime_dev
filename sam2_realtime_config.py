"""Runtime defaults for the ROS2 SAM2 realtime demo.

The values in SAM2_REALTIME_PROFILE are the default parameters used by
`demo_multi_point_topic.py` when it is launched simply as:

    python demo_multi_point_topic.py

They can still be overridden at runtime with environment variables, which is
useful for experiments without editing code:

    SAM2_IMAGE_SIZE=1024 python demo_multi_point_topic.py
    SAM2_NUM_MASKMEM=7 python demo_multi_point_topic.py
    SAM2_MEMORY_TEMPORAL_STRIDE=3 python demo_multi_point_topic.py

Important implementation detail:
- `image_size`, `compile_image_encoder`, and `memory_temporal_stride_for_eval`
  are applied as Hydra overrides before model construction.
- `runtime_num_maskmem` is applied after checkpoint loading, because changing
  `model.num_maskmem` before loading would change the checkpoint tensor shape of
  `maskmem_tpos_enc` and cause a load_state_dict size mismatch.
"""

SAM2_REALTIME_PROFILE = {
    "name": "sam2.1-small-realtime-768-maskmem5-stride2",

    # Model selection
    "checkpoint": "./checkpoints/sam2.1_hiera_small.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",

    # Keep the PyTorch-compiled image encoder enabled by default. In this fork,
    # sam2/modeling/sam2_base.py should compile it with
    # mode="max-autotune-no-cudagraphs" to avoid CUDA Graph lifetime issues.
    "compile_image_encoder": True,

    # Internal SAM2 image resolution. 768 gave a strong speedup while preserving
    # more detail than more aggressive reductions such as 512.
    "image_size": 768,

    # Runtime temporal memory window. The checkpoint is loaded with the original
    # num_maskmem=7 and then reduced to this value after loading.
    "runtime_num_maskmem": 5,

    # Temporal stride used to select memory frames during evaluation/tracking.
    # stride=2 is a conservative speed/robustness compromise; stride=3 was faster
    # in FPS tests but may skip too much recent temporal context in moving scenes.
    "memory_temporal_stride_for_eval": 2,
}
