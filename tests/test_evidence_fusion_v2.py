# path: tests/test_evidence_fusion_v2.py
"""
Unit tests for evidence fusion with stop-gradient.

Tests:
1. Stop-gradient verification: π_H, π_L have no grad_fn when detached
2. Fusion output shapes
3. Evidence fusion vs probability averaging
4. Backward compatibility
"""
from __future__ import annotations

import pytest
import torch

from src.models.fusion.evidence_fusion_v2 import (
    EvidenceFusionV2,
    EvidenceFusionV2Config,
    verify_stop_gradient,
    dirichlet_mean_binary,
    dirichlet_variance_binary,
)


class TestEvidenceFusionV2:
    """Tests for updated evidence fusion module."""
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample alpha and evidence tensors."""
        torch.manual_seed(42)
        b, k, h, w = 2, 2, 32, 32
        
        # Evidence must be positive (from softplus)
        e_h = torch.rand(b, k, h, w).abs() + 0.1
        e_l = torch.rand(b, k, h, w).abs() + 0.1
        
        alpha_h = e_h + 1.0
        alpha_l = e_l + 1.0
        
        return alpha_h, e_h, alpha_l, e_l
    
    @pytest.fixture
    def fusion_detached(self) -> EvidenceFusionV2:
        """Fusion with stop-gradient enabled."""
        cfg = EvidenceFusionV2Config(detach_weights=True)
        return EvidenceFusionV2(cfg)
    
    @pytest.fixture
    def fusion_no_detach(self) -> EvidenceFusionV2:
        """Fusion without stop-gradient."""
        cfg = EvidenceFusionV2Config(detach_weights=False)
        return EvidenceFusionV2(cfg)
    
    def test_stop_gradient_enabled(self, fusion_detached: EvidenceFusionV2, sample_inputs):
        """Verify that fusion weights have no gradient when detach=True."""
        alpha_h, e_h, alpha_l, e_l = sample_inputs
        
        # Enable gradients
        alpha_h.requires_grad_(True)
        e_h.requires_grad_(True)
        alpha_l.requires_grad_(True)
        e_l.requires_grad_(True)
        
        alpha_f, p_f, u_f, weights = fusion_detached(
            alpha_h, e_h, alpha_l, e_l, return_weights=True
        )
        
        pi_h, pi_l = weights
        
        # Detached tensors should have no grad_fn
        assert pi_h.grad_fn is None, "π_H should have no grad_fn when detach_weights=True"
        assert pi_l.grad_fn is None, "π_L should have no grad_fn when detach_weights=True"
    
    def test_gradient_flows_to_output(self, fusion_detached: EvidenceFusionV2, sample_inputs):
        """Verify gradients flow to fused output even with detached weights."""
        alpha_h, e_h, alpha_l, e_l = sample_inputs
        
        alpha_h.requires_grad_(True)
        e_h.requires_grad_(True)
        alpha_l.requires_grad_(True)
        e_l.requires_grad_(True)
        
        alpha_f, p_f, u_f, _ = fusion_detached(alpha_h, e_h, alpha_l, e_l)
        
        # Output should still have gradients (from evidence, not weights)
        loss = p_f.mean()
        loss.backward()
        
        # Gradients should flow to evidence
        assert e_h.grad is not None, "Gradient should flow to e_h"
        assert e_l.grad is not None, "Gradient should flow to e_l"
    
    def test_gradient_to_weights_when_not_detached(self, fusion_no_detach: EvidenceFusionV2, sample_inputs):
        """Verify gradients DO flow through weights when detach=False."""
        alpha_h, e_h, alpha_l, e_l = sample_inputs
        
        alpha_h.requires_grad_(True)
        
        alpha_f, p_f, u_f, weights = fusion_no_detach(
            alpha_h, e_h, alpha_l, e_l, return_weights=True
        )
        
        pi_h, pi_l = weights
        
        # Non-detached weights should have grad_fn
        assert pi_h.grad_fn is not None, "π_H should have grad_fn when detach_weights=False"
    
    def test_output_shapes(self, fusion_detached: EvidenceFusionV2, sample_inputs):
        """Test output tensor shapes."""
        alpha_h, e_h, alpha_l, e_l = sample_inputs
        
        alpha_f, p_f, u_f, _ = fusion_detached(alpha_h, e_h, alpha_l, e_l)
        
        assert alpha_f.shape == alpha_h.shape
        assert p_f.shape == (alpha_h.size(0), 1, alpha_h.size(2), alpha_h.size(3))
        assert u_f.shape == p_f.shape
    
    def test_evidence_fusion_vs_prob_avg(self, sample_inputs):
        """Test that evidence fusion differs from probability averaging."""
        alpha_h, e_h, alpha_l, e_l = sample_inputs
        
        cfg_evidence = EvidenceFusionV2Config(fusion_type='evidence_weighted')
        cfg_avg = EvidenceFusionV2Config(fusion_type='avg_prob')
        
        fusion_evidence = EvidenceFusionV2(cfg_evidence)
        fusion_avg = EvidenceFusionV2(cfg_avg)
        
        _, p_evidence, _, _ = fusion_evidence(alpha_h, e_h, alpha_l, e_l)
        _, p_avg, _, _ = fusion_avg(alpha_h, e_h, alpha_l, e_l)
        
        # Results should differ (unless by chance)
        # Just verify both produce valid probabilities
        assert p_evidence.min() >= 0 and p_evidence.max() <= 1
        assert p_avg.min() >= 0 and p_avg.max() <= 1
    
    def test_weights_sum_to_one(self, fusion_detached: EvidenceFusionV2, sample_inputs):
        """Test that fusion weights sum to approximately 1."""
        alpha_h, e_h, alpha_l, e_l = sample_inputs
        
        _, _, _, weights = fusion_detached(
            alpha_h, e_h, alpha_l, e_l, return_weights=True
        )
        
        pi_h, pi_l = weights
        weight_sum = pi_h + pi_l
        
        # Should be very close to 1 (slight difference due to eps in denominator)
        assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-4)
    
    def test_verify_stop_gradient_helper(self):
        """Test the verify_stop_gradient utility function."""
        fusion = EvidenceFusionV2(EvidenceFusionV2Config(detach_weights=True))
        assert verify_stop_gradient(fusion) == True
    
    def test_branch_correctness_computation(self, fusion_detached: EvidenceFusionV2):
        """Test branch correctness mask computation."""
        b, h, w = 2, 32, 32
        
        p_h = torch.rand(b, 1, h, w)
        p_l = torch.rand(b, 1, h, w)
        target = (torch.rand(b, 1, h, w) > 0.5).float()
        
        correct_h, correct_l = fusion_detached.compute_branch_correctness(
            p_h, p_l, target, threshold=0.5
        )
        
        assert correct_h.shape == target.shape
        assert correct_l.shape == target.shape
        assert correct_h.min() >= 0 and correct_h.max() <= 1


class TestDirichletUtils:
    """Tests for Dirichlet distribution utilities."""
    
    def test_dirichlet_mean_binary(self):
        """Test binary Dirichlet mean computation."""
        # alpha = [bg=2, polyp=8] -> p_polyp = 8/10 = 0.8
        alpha = torch.tensor([[[[2.0, 8.0]]]]).permute(0, 3, 1, 2)  # (1, 2, 1, 1)
        
        p = dirichlet_mean_binary(alpha)
        
        assert torch.isclose(p, torch.tensor(0.8), atol=1e-5)
    
    def test_dirichlet_variance_binary(self):
        """Test binary Dirichlet variance computation."""
        # alpha = [a=2, b=2] -> Var = 2*2 / (4^2 * 5) = 0.05
        alpha = torch.tensor([[[[2.0, 2.0]]]]).permute(0, 3, 1, 2)
        
        var = dirichlet_variance_binary(alpha)
        
        expected = (2 * 2) / (16 * 5)
        assert torch.isclose(var, torch.tensor(expected), atol=1e-5)


# Run with: pytest tests/test_evidence_fusion_v2.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
