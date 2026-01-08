# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import torch

import kornia
from kornia.contrib import DistanceTransform, distance_transform
from testing.base import BaseTester


class TestConvDistanceTransform(BaseTester):
    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_smoke(self, kernel_size, device, dtype):
        # 2D case
        sample2d = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)
        output2d = distance_transform(sample2d, kernel_size)
        assert isinstance(output2d, torch.Tensor)
        assert output2d.shape == sample2d.shape

        # 3D case
        sample3d = torch.rand(1, 1, 3, 10, 10, device=device, dtype=dtype)
        output3d = distance_transform(sample3d, kernel_size)
        assert isinstance(output3d, torch.Tensor)
        assert output3d.shape == sample3d.shape

    def test_exception(self, device, dtype):
        sample = torch.rand(1, 1, 10, 10, device=device, dtype=dtype)

        # Non-odd kernel size
        with pytest.raises(ValueError):
            distance_transform(sample, kernel_size=4)

        # Invalid input dimensions (too few)
        with pytest.raises(ValueError):
            distance_transform(torch.rand(10, 10, device=device, dtype=dtype))

        # Invalid input type
        with pytest.raises(TypeError):
            distance_transform(None)

    def test_cardinality(self, device, dtype):
        # Test 2D
        sample2d = torch.rand(2, 3, 50, 50, device=device, dtype=dtype)
        out2d = distance_transform(sample2d)
        assert out2d.shape == (2, 3, 50, 50)

        # Test 3D
        sample3d = torch.rand(2, 3, 5, 20, 20, device=device, dtype=dtype)
        out3d = distance_transform(sample3d)
        assert out3d.shape == (2, 3, 5, 20, 20)

    def test_value(self, device, dtype):
        # Euclidean Verification
        B, C, H, W = 1, 1, 4, 4
        kernel_size = 7
        h = 0.35
        sample1 = torch.zeros(B, C, H, W, device=device, dtype=dtype)
        # Point at (1, 1)
        sample1[:, :, 1, 1] = 1.0
        
        # Expected Euclidean distances from (1,1)
        # (0,0)->sqrt(2), (0,1)->1, (0,2)->sqrt(2), (0,3)->sqrt(5)
        # (1,0)->1,       (1,1)->0, (1,2)->1,       (1,3)->2
        # ... etc
        expected_output1 = torch.tensor(
            [
                [
                    [
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [1.0000000000, 0.0000000000, 1.0000000000, 2.0000000000],
                        [1.4142135382, 1.0000000000, 1.4142135382, 2.2360680103],
                        [2.2360680103, 2.0000000000, 2.2360680103, 2.8284270763],
                    ]
                ]
            ],
            device=device,
            dtype=dtype,
        )
        output1 = distance_transform(sample1, kernel_size, h)
        self.assert_close(expected_output1, output1)

    def test_offset_parenthesis_fix(self, device, dtype):
        # Regression test for offset calculation, valid for Euclidean as well
        # (Distance to a horizontal line of ones is just vertical distance)
        img = torch.zeros(1, 1, 8, 4, device=device, dtype=dtype)
        img[0, 0, 1, :] = 1.0
        out = distance_transform(img, kernel_size=3, h=0.01)
        expected = torch.tensor(
            [
                [0.9998, 0.9998, 0.9998, 0.9998],
                [0.0000, 0.0000, 0.0000, 0.0000],
                [0.9998, 0.9998, 0.9998, 0.9998],
                [1.9998, 1.9998, 1.9998, 1.9998],
                [2.9998, 2.9998, 2.9998, 2.9998],
                [3.9998, 3.9998, 3.9998, 3.9998],
                [4.9998, 4.9998, 4.9998, 4.9998],
                [5.9998, 5.9998, 5.9998, 5.9998],
            ],
            device=device,
            dtype=dtype,
        )

        self.assert_close(out[0, 0], expected, rtol=1e-3, atol=1e-3)

    def test_module(self, device, dtype):
        B, C, H, W = 1, 1, 50, 50
        sample1 = torch.rand(B, C, H, W, device=device, dtype=dtype)
        dt_module = DistanceTransform().to(device, dtype)

        output1 = dt_module(sample1)
        output2 = distance_transform(sample1)
        self.assert_close(output1, output2)

    def test_gradcheck(self, device):
        B, C, H, W = 1, 1, 10, 10
        sample1 = torch.ones(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        self.gradcheck(distance_transform, (sample1,))

    def test_dynamo(self, device, dtype, torch_optimizer):
        sample = torch.rand(1, 1, 20, 20, device=device, dtype=dtype)
        op = distance_transform
        op_optimized = torch_optimizer(op)
        self.assert_close(op(sample), op_optimized(sample))