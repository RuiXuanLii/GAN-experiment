import pytest
import torch
from scripts.export_images import to_uint8
from scripts.freeze_protocol import freeze


def test_image_conversion_exact_endpoints_channels_and_dtype():
    x=torch.tensor([-1.,0.,1.]).reshape(1,3,1,1)
    y=to_uint8(x)
    assert y.shape==(1,1,1,3) and str(y.dtype)=='uint8'
    assert y.tolist()==[[[[0,128,255]]]]


def test_image_conversion_rejects_nonfinite():
    with pytest.raises(ValueError): to_uint8(torch.full((1,3,2,2),float('nan')))


def test_freeze_refuses_without_actual_gpu_pilots(tmp_path):
    with pytest.raises(ValueError,match='Four completed GPU pilots'):
        freeze(tmp_path,tmp_path,tmp_path/'freeze.json',gpu_hours=10)
