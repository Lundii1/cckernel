import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)
