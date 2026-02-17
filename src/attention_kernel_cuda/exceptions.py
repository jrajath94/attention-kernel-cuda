"""Custom exceptions for attention kernel operations."""


class AttentionKernelError(Exception):
    """Base exception for attention kernel errors."""


class InvalidHeadDimensionError(AttentionKernelError):
    """Raised when the head dimension is unsupported."""

    def __init__(self, head_dim: int, max_dim: int = 512) -> None:
        self.head_dim = head_dim
        self.max_dim = max_dim
        super().__init__(
            f"Head dimension {head_dim} is invalid. Must be in range [1, {max_dim}]."
        )


class TilingConfigError(AttentionKernelError):
    """Raised when tiling configuration is invalid for given dimensions."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Tiling configuration error: {message}")


class CUDANotAvailableError(AttentionKernelError):
    """Raised when CUDA is required but not available."""

    def __init__(self) -> None:
        super().__init__(
            "CUDA is not available. Install PyTorch with CUDA support "
            "or use the reference CPU implementation."
        )


class SharedMemoryExceededError(AttentionKernelError):
    """Raised when kernel requires more shared memory than available."""

    def __init__(self, required: int, available: int) -> None:
        self.required = required
        self.available = available
        super().__init__(
            f"Kernel requires {required} bytes of shared memory, "
            f"but only {available} bytes available on this GPU."
        )
