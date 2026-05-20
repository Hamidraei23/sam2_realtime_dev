"""Runtime defaults for the ROS2 SAM2 realtime demo.

The values in SAM2_REALTIME_PROFILE are the default parameters used by
`demo_multi_point_topic.py` when it is launched simply as:

    python demo_multi_point_topic.py

They can still be overridden at runtime with environment variables, which is
useful for experiments without editing code:

    SAM2_IMAGE_SIZE=1024 python demo_multi_point_topic.py
    SAM2_NUM_MASKMEM=7 python demo_multi_point_topic.py
    SAM2_MEMORY_TEMPORAL_STRIDE=3 python demo_multi_point_topic.py
    SAM2_COMPILE_IMAGE_ENCODER=0 python demo_multi_point_topic.py
    SAM2_COMPILE_IMAGE_ENCODER_FULLGRAPH=0 python demo_multi_point_topic.py
    SAM2_COMPILE_MEMORY_ATTENTION=0 python demo_multi_point_topic.py
    SAM2_COMPILE_MEMORY_ATTENTION_FULLGRAPH=1 python demo_multi_point_topic.py
    SAM2_COMPILE_MEMORY_ATTENTION_DYNAMIC=1 python demo_multi_point_topic.py
    SAM2_PROFILE_TIMERS=1 python demo_multi_point_topic.py
    SAM2_SHOW_FPS_OVERLAY=1 python demo_multi_point_topic.py
    SAM2_FPS_OVERLAY_WINDOW=5 python demo_multi_point_topic.py
    SAM2_COMPILE_MEMORY_ENCODER=0 python demo_multi_point_topic.py
    SAM2_COMPILE_SAM_MASK_DECODER=0 python demo_multi_point_topic.py

Important implementation detail:
- `image_size`, `compile_image_encoder`, `compile_memory_attention`,
  `compile_memory_encoder`, `compile_sam_mask_decoder`, and
  `memory_temporal_stride_for_eval` are applied as Hydra overrides before model
  construction.
- `runtime_num_maskmem` is applied after checkpoint loading, because changing
  `model.num_maskmem` before loading would change the checkpoint tensor shape of
  `maskmem_tpos_enc` and cause a load_state_dict size mismatch.
"""

SAM2_REALTIME_PROFILE = {
    "name": "sam2.1-small-realtime-768-maskmem5-stride2",

    # Demo/profiling defaults. Timers are disabled by default for normal runs
    # because they add synchronization/formatting overhead and produce verbose
    # output. Enable them only for profiling with SAM2_PROFILE_TIMERS=1.
    "profile_timers": False,
    "profile_sync_cuda": True,
    "profile_print_every": 60,

    # Optional visual FPS marker drawn in the top-right corner of the displayed
    # segmentation image. Disabled by default to keep the visualization clean.
    # Enable it with SAM2_SHOW_FPS_OVERLAY=1. The value shown is the rolling
    # average over the last `fps_overlay_window` processed frames.
    "show_fps_overlay": False,
    "fps_overlay_window": 3,

    # Model selection
    "checkpoint": "./checkpoints/sam2.1_hiera_small.pt",
    "model_cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",

    # Keep the PyTorch-compiled image encoder enabled by default. It is handled
    # by the same centralized compile block as the other optional compiled
    # modules in sam2/modeling/sam2_base.py. The mode disables CUDA Graphs to
    # avoid lifetime/reuse issues with cached positional encodings in the
    # realtime camera loop.
    "compile_image_encoder": True,
    "compile_image_encoder_fullgraph": True,
    "compile_image_encoder_dynamic": False,
    "compile_image_encoder_mode": "max-autotune-no-cudagraphs",

    # Compile the memory-attention module, which is the dominant bottleneck in
    # multi-object tracking. This is intentionally configurable because the first
    # few tracking frames can be slower while TorchInductor compiles shape-specific
    # graphs. Set SAM2_COMPILE_MEMORY_ATTENTION=0 to disable it.
    "compile_memory_attention": True,

    # Safer default for memory attention than fullgraph=True. RoPEAttention updates
    # cached rotary encodings internally, so allowing graph breaks is more robust
    # while still letting TorchInductor optimize the heavy tensor kernels.
    "compile_memory_attention_fullgraph": False,
    "compile_memory_attention_dynamic": False,
    "compile_memory_attention_mode": "max-autotune-no-cudagraphs",

    # Compile the memory encoder. This is a smaller block than memory attention,
    # but it scales with the number of tracked objects. Set
    # SAM2_COMPILE_MEMORY_ENCODER=0 to disable it.
    "compile_memory_encoder": True,
    "compile_memory_encoder_fullgraph": False,
    "compile_memory_encoder_dynamic": False,
    "compile_memory_encoder_mode": "max-autotune-no-cudagraphs",

    # Compile the SAM mask decoder. This block is relatively small, so the gain
    # may be modest, but it is measurable with several tracked objects. Set
    # SAM2_COMPILE_SAM_MASK_DECODER=0 to disable it.
    "compile_sam_mask_decoder": True,
    "compile_sam_mask_decoder_fullgraph": False,
    "compile_sam_mask_decoder_dynamic": False,
    "compile_sam_mask_decoder_mode": "max-autotune-no-cudagraphs",

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
