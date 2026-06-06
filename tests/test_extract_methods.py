"""Unit tests for the ExtractMethods class."""

import pytest
import torch

from altx import ExtractMethods


class TestExcessKurtosis:
    """Tests for ExtractMethods.excess_kurtosis."""

    def test_known_values(self) -> None:
        """Excess kurtosis of [1,2,3] is -1.5."""
        # Shape (1, 3, 2): one percentile level, 3 windows, 2 channels.
        # Both channels: [1,2,3] and [4,5,6] — same deviations, same kurtosis.
        t = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])
        result = ExtractMethods.excess_kurtosis(t)
        # mean=[2,5], deviations=[-1,0,1], var=2/3, 4th_moment=2/3
        # kurt=(2/3)/(2/3)^2=3/2, excess=3/2-3=-3/2
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.tensor([[-1.5, -1.5]]), atol=1e-5)

    def test_constant_input_returns_zeros(self) -> None:
        """Constant values produce zero variance, so zeros are returned."""
        t = torch.ones(1, 5, 2)
        result = ExtractMethods.excess_kurtosis(t)
        assert result.shape == (1, 2)
        assert torch.all(result == 0)

    def test_output_shape_single_percentile_level(self) -> None:
        """Shape (1, n, m) — the only valid call pattern — yields (1, m)."""
        # excess_kurtosis is always called with a single percentile slice,
        # so the first dimension is always 1.
        torch.manual_seed(99)
        t = torch.randn(1, 10, 4).abs() + 0.1
        result = ExtractMethods.excess_kurtosis(t)
        assert result.shape == (1, 4)


class TestNthMoment:
    """Tests for ExtractMethods.nth_moment."""

    def test_first_moment_is_zero(self) -> None:
        """First central moment (mean of deviations) is always zero."""
        torch.manual_seed(0)
        t = torch.randn(1, 10, 3)
        result = ExtractMethods.nth_moment(t, n=1)
        assert result.shape == (1, 3)
        assert torch.allclose(result, torch.zeros(1, 3), atol=1e-6)

    def test_second_moment_known_values(self) -> None:
        """Second central moment of [1,2,3] is 2/3 (biased variance)."""
        t = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])
        result = ExtractMethods.nth_moment(t, n=2)
        expected = torch.tensor([[2.0 / 3.0, 2.0 / 3.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_default_n_equals_four(self) -> None:
        """Default n=4 matches explicit nth_moment(t, n=4)."""
        torch.manual_seed(1)
        t = torch.randn(1, 8, 2)
        assert torch.allclose(
            ExtractMethods.nth_moment(t), ExtractMethods.nth_moment(t, n=4)
        )

    def test_nan_input_returns_zeros(self) -> None:
        """NaN values in input trigger zeros fallback."""
        t = torch.full((1, 3, 1), float("nan"))
        result = ExtractMethods.nth_moment(t, n=2)
        assert result.shape == (1, 1)
        assert torch.all(result == 0)

    def test_constant_input_returns_zeros(self) -> None:
        """Constant values give zero moment for n>=2."""
        t = torch.ones(1, 5, 2)
        result = ExtractMethods.nth_moment(t, n=2)
        assert torch.allclose(result, torch.zeros(1, 2), atol=1e-6)


class TestExtract:
    """Tests for ExtractMethods.extract."""

    def test_mean_method_constant_input(self) -> None:
        """Mean of constant F=1 after squaring is 1."""
        F = torch.ones(3, 4, 2)
        result = ExtractMethods.extract(F, [["mean", 0.5]])
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.ones(1, 2))

    def test_var_method_constant_input(self) -> None:
        """Variance of constant signal is zero."""
        F = torch.ones(3, 4, 2)
        result = ExtractMethods.extract(F, [["var", 0.5]])
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.zeros(1, 2))

    def test_mean_all_method(self) -> None:
        """mean_all returns mean of F^2 across all dims."""
        F = torch.full((3, 4, 2), 2.0)
        result = ExtractMethods.extract(F, [["mean_all", None]])
        # F^2 = 4 everywhere, mean = 4
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.full((1, 2), 4.0))

    def test_fourth_moment_constant_is_zero(self) -> None:
        """4th moment of a constant signal is zero."""
        F = torch.ones(3, 4, 2)
        result = ExtractMethods.extract(F, [["4th_moment", 0.5]])
        assert result.shape == (1, 2)
        assert torch.allclose(result, torch.zeros(1, 2), atol=1e-6)

    def test_excess_kurtosis_method(self) -> None:
        """excess_kurtosis method runs without error and returns (1, m)."""
        torch.manual_seed(2)
        F = torch.randn(5, 8, 2).abs() + 0.1
        result = ExtractMethods.extract(F, [["excess_kurtosis", 0.5]])
        assert result.shape == (1, 2)

    def test_invalid_method_raises_value_error(self) -> None:
        """Unsupported method name raises ValueError."""
        F = torch.ones(3, 4, 2)
        with pytest.raises(ValueError, match="not implemented"):
            ExtractMethods.extract(F, [["not_a_method", 0.5]])

    def test_multiple_methods_stacked(self) -> None:
        """Output shape is (n_methods, m) when multiple methods are used."""
        F = torch.ones(3, 4, 2)
        extr_methods = [["mean", 0.5], ["var", 0.5], ["mean_all", None]]
        result = ExtractMethods.extract(F, extr_methods)
        assert result.shape == (3, 2)

    def test_two_different_percentiles(self) -> None:
        """Two methods with distinct percentiles are handled correctly."""
        torch.manual_seed(3)
        F = torch.rand(4, 6, 2)
        extr_methods = [["mean", 0.1], ["mean", 0.9]]
        result = ExtractMethods.extract(F, extr_methods)
        assert result.shape == (2, 2)

    def test_device_string_cpu(self) -> None:
        """Passing device='cpu' as a string is accepted."""
        F = torch.ones(3, 4, 2)
        result = ExtractMethods.extract(F, [["mean", 0.5]], device="cpu")
        assert result.shape == (1, 2)

    def test_squaring_is_applied_before_quantile(self) -> None:
        """F is squared internally, so negative values map to positive features."""
        F = torch.full((3, 4, 2), -2.0)
        result = ExtractMethods.extract(F, [["mean_all", None]])
        # (-2)^2 = 4
        assert torch.allclose(result, torch.full((1, 2), 4.0))
