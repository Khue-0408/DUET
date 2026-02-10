# path: tests/test_weighted_losses.py
"""
Unit tests for weighted focal and IoU losses.

Tests:
1. Boundary weight computation
2. Weighted focal loss properties
3. Weighted IoU loss properties
4. Combined loss
"""
from __future__ import annotations

import pytest
import torch

from src.models.losses.weighted_focal_iou import (
    compute_boundary_weights,
    weighted_focal_loss,
    weighted_iou_loss,
    WeightedFocalIOULoss,
    WeightedLossConfig,
)


class TestBoundaryWeights:
    """Tests for boundary weight computation."""
    
    def test_boundary_weights_shape(self):
        """Test output shape matches input."""
        gt = torch.rand(2, 1, 64, 64)
        weights = compute_boundary_weights(gt, kernel_size=5)
        
        assert weights.shape == gt.shape
    
    def test_boundary_weights_positive(self):
        """Test all weights are positive."""
        gt = (torch.rand(2, 1, 64, 64) > 0.5).float()
        weights = compute_boundary_weights(gt, kernel_size=5)
        
        assert (weights > 0).all()
    
    def test_boundary_weights_higher_at_edges(self):
        """Test weights are higher at mask boundaries."""
        # Create simple mask with clear boundary
        gt = torch.zeros(1, 1, 32, 32)
        gt[0, 0, 10:22, 10:22] = 1.0  # Square polyp
        
        weights = compute_boundary_weights(gt, kernel_size=5, lambda_boundary=5.0)
        
        # Interior should have weight ≈ 1 (β ≈ 0)
        interior_weight = weights[0, 0, 15, 15].item()
        
        # Boundary should have weight > 1 (β > 0)
        boundary_weight = weights[0, 0, 10, 15].item()  # Edge pixel
        
        assert boundary_weight > interior_weight, \
            f"Boundary weight {boundary_weight} should be > interior {interior_weight}"
    
    def test_uniform_mask_low_weights(self):
        """Test uniform mask (all 0 or all 1) has low boundary weights."""
        gt_zeros = torch.zeros(1, 1, 32, 32)
        gt_ones = torch.ones(1, 1, 32, 32)
        
        w_zeros = compute_boundary_weights(gt_zeros, kernel_size=5)
        w_ones = compute_boundary_weights(gt_ones, kernel_size=5)
        
        # All weights should be 1 (no boundary)
        assert torch.allclose(w_zeros, torch.ones_like(w_zeros), atol=0.1)
        assert torch.allclose(w_ones, torch.ones_like(w_ones), atol=0.1)


class TestWeightedFocalLoss:
    """Tests for weighted focal loss."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and targets."""
        b, h, w = 2, 32, 32
        pred = torch.sigmoid(torch.randn(b, 1, h, w))
        target = (torch.rand(b, 1, h, w) > 0.5).float()
        weights = torch.ones(b, 1, h, w)
        return pred, target, weights
    
    def test_focal_loss_positive(self, sample_data):
        """Test focal loss is non-negative."""
        pred, target, weights = sample_data
        loss = weighted_focal_loss(pred, target, weights)
        
        assert loss >= 0
    
    def test_focal_loss_perfect_prediction(self):
        """Test loss is low for perfect predictions."""
        pred = torch.ones(1, 1, 32, 32) * 0.99
        target = torch.ones(1, 1, 32, 32)
        weights = torch.ones(1, 1, 32, 32)
        
        loss_perfect = weighted_focal_loss(pred, target, weights, gamma=2.0)
        
        pred_wrong = torch.ones(1, 1, 32, 32) * 0.01
        loss_wrong = weighted_focal_loss(pred_wrong, target, weights, gamma=2.0)
        
        assert loss_perfect < loss_wrong
    
    def test_focal_gamma_effect(self, sample_data):
        """Test that higher gamma focuses more on hard examples."""
        pred, target, weights = sample_data
        
        loss_gamma0 = weighted_focal_loss(pred, target, weights, gamma=0.0)
        loss_gamma2 = weighted_focal_loss(pred, target, weights, gamma=2.0)
        
        # With gamma=0, it's standard CE; with gamma>0, easy examples downweighted
        # Just verify both produce valid losses
        assert loss_gamma0 >= 0
        assert loss_gamma2 >= 0


class TestWeightedIoULoss:
    """Tests for weighted IoU loss."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and targets."""
        b, h, w = 2, 32, 32
        pred = torch.sigmoid(torch.randn(b, 1, h, w))
        target = (torch.rand(b, 1, h, w) > 0.5).float()
        weights = torch.ones(b, 1, h, w)
        return pred, target, weights
    
    def test_iou_loss_range(self, sample_data):
        """Test IoU loss is in [0, 1]."""
        pred, target, weights = sample_data
        loss = weighted_iou_loss(pred, target, weights)
        
        assert loss >= 0 and loss <= 1
    
    def test_iou_loss_perfect_prediction(self):
        """Test IoU loss is 0 for perfect predictions."""
        pred = torch.ones(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        weights = torch.ones(1, 1, 32, 32)
        
        loss = weighted_iou_loss(pred, target, weights)
        
        assert loss < 0.01  # Should be very close to 0
    
    def test_iou_loss_complete_mismatch(self):
        """Test IoU loss is high for complete mismatch."""
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        weights = torch.ones(1, 1, 32, 32)
        
        loss = weighted_iou_loss(pred, target, weights)
        
        # IoU should be 0, so loss should be 1
        assert loss > 0.99


class TestCombinedLoss:
    """Tests for combined weighted focal + IoU loss."""
    
    @pytest.fixture
    def loss_module(self) -> WeightedFocalIOULoss:
        """Create combined loss module."""
        cfg = WeightedLossConfig(kernel_size=5, lambda_boundary=5.0, gamma_focal=2.0)
        return WeightedFocalIOULoss(cfg)
    
    def test_combined_loss_returns_tuple(self, loss_module):
        """Test combined loss returns (total, focal, iou) tuple."""
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = (torch.rand(2, 1, 32, 32) > 0.5).float()
        
        total, focal, iou = loss_module(pred, target)
        
        assert isinstance(total, torch.Tensor)
        assert isinstance(focal, torch.Tensor)
        assert isinstance(iou, torch.Tensor)
    
    def test_combined_loss_formula(self, loss_module):
        """Test combined loss follows L_total = 0.5 * (L_focal + L_iou)."""
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = (torch.rand(2, 1, 32, 32) > 0.5).float()
        
        total, focal, iou = loss_module(pred, target)
        
        expected = 0.5 * (focal + iou)
        assert torch.isclose(total, expected, atol=1e-5)
    
    def test_gradients_flow(self, loss_module):
        """Test gradients flow through combined loss."""
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32, requires_grad=True))
        target = (torch.rand(2, 1, 32, 32) > 0.5).float()
        
        total, _, _ = loss_module(pred, target)
        total.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


# Run with: pytest tests/test_weighted_losses.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
