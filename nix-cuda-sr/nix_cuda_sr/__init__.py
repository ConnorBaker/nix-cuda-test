__version__ = "0.1.0"


def _setup() -> None:
    import warnings  # noqa: PLC0415

    import cutlass  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import torch._dynamo.config  # noqa: PLC0415
    import torch._inductor.config  # noqa: PLC0415
    import torch.backends.cuda  # noqa: PLC0415
    import torch.backends.cudnn  # noqa: PLC0415
    from lightning_fabric.fabric import Fabric  # noqa: PLC0415

    Fabric.seed_everything(42, workers=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark_limit = 100

    # BF16 should be enough for our use case.
    # See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("medium")  # type: ignore

    torch._inductor.config.dce = True
    torch._inductor.config.enable_auto_functionalized_v2 = True
    torch._inductor.config.memory_planning = True
    torch._inductor.config.permute_fusion = True
    torch._inductor.config.b2b_gemm_pass = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_pointwise = True
    torch._inductor.config.max_autotune_gemm_backends = "ATEN,TRITON,CUTLASS"
    torch._inductor.config.max_autotune_conv_backends = "ATEN,TRITON,CUTLASS"
    # torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"
    torch._inductor.config.max_autotune_gemm = True
    torch._inductor.config.force_layout_optimization = True
    torch._inductor.config.warn_mix_layout = True
    torch._inductor.config.pad_channels_last = True
    torch._inductor.config.use_minimal_arrayref_interface = True
    torch._inductor.config.decompose_mem_bound_mm = True
    torch._inductor.config.assume_aligned_inputs = True

    torch._inductor.config.combo_kernels = True
    torch._inductor.config.benchmark_combo_kernel = True
    torch._inductor.config.combo_kernels_autotune = 2
    torch._inductor.config.combo_kernel_allow_mixed_sizes = 2
    torch._inductor.config.combo_kernel_foreach_dynamic_shapes = True

    torch._inductor.config.permute_fusion = True
    torch._inductor.config.size_asserts = False

    torch._inductor.config.triton.autotune_at_compile_time = True
    torch._inductor.config.triton.multi_kernel = True
    # torch._inductor.config.triton.use_block_ptr = True  # Causes a compile error

    torch._inductor.config.cuda.arch = "89"
    torch._inductor.config.cuda.version = "12.8"
    torch._inductor.config.cuda.compile_opt_level = "-O3"
    torch._inductor.config.cuda.enable_cuda_lto = True
    torch._inductor.config.cuda.use_fast_math = True

    # TODO(@connorbaker):
    # W0427 10:19:54.654000 810187 torch/_inductor/utils.py:1115] [0/0] Failed to import CUTLASS lib. Please check
    # whether _inductor.config.cuda.cutlass_dir is set correctly. Skipping CUTLASS backend for now.
    # TODO(@connorbaker): Patch the PyTorch build to replace the default value with the one for cutlass src in the Nix
    # store.
    torch._inductor.config.cuda.cutlass_dir = cutlass.source_path

    torch._dynamo.config.cache_size_limit = 8192
    torch._dynamo.config.accumulated_cache_size_limit = 65536
    # torch._dynamo.reset()  # type: ignore[no-untyped-call]

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # te_attention._log_level = 2
    # te_attention.fa_logger.setLevel(logging.DEBUG)
