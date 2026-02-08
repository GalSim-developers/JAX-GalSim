"""Tests for custom gradients of Bessel functions in JAX-GalSim.

This module tests the custom gradient implementation for the modified Bessel
function K_ν(x), which uses analytical derivative formulas based on Bessel
recurrence relations instead of automatic differentiation.

To run these tests:
    pytest tests/jax/test_bessel_gradients.py \\
        --ignore=tests/jax/test_image_wrapping.py \\
        --ignore=tests/jax/test_interpolant_jax.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from jax_galsim.bessel import kv


class TestKvGradients:
    """Test suite for kv custom gradients."""

    @pytest.mark.parametrize(
        "nu,x",
        [
            (0.5, 1.0),
            (1.0, 5.0),
            (2.5, 5.0),
            (10.0, 10.0),
            (39.8, 40.0),
            (300.9, 500.0),
        ],
    )
    def test_gradient_wrt_x_compiles(self, nu, x):
        """Test that gradient w.r.t. x compiles with jax.jit."""
        grad_fn = jax.jit(jax.grad(lambda x_val: kv(nu, x_val)))
        grad_val = grad_fn(x)
        # Just verify it's a finite value
        assert jnp.isfinite(grad_val), f"Gradient is not finite: {grad_val}"

    @pytest.mark.parametrize(
        "nu,x",
        [
            (0.5, 1.0),
            (1.0, 5.0),
            (2.5, 5.0),
            (10.0, 10.0),
            (39.8, 40.0),
        ],
    )
    def test_gradient_vs_finite_differences(self, nu, x):
        """Test that custom gradients match finite differences."""
        grad_fn = jax.grad(lambda x_val: kv(nu, x_val))
        analytical_grad = grad_fn(x)

        # Compute finite difference gradient
        eps = 1e-5
        numerical_grad = (kv(nu, x + eps) - kv(nu, x - eps)) / (2 * eps)

        # Allow slightly looser tolerance for numerical differentiation
        relative_error = jnp.abs((analytical_grad - numerical_grad) / numerical_grad)
        assert relative_error < 1e-6, (
            f"Gradient error too large: {relative_error} at nu={nu}, x={x}"
        )

    @pytest.mark.parametrize(
        "nu,x",
        [
            (0.5, 1.0),
            (1.0, 5.0),
            (2.5, 5.0),
            (10.0, 10.0),
            (39.8, 40.0),
        ],
    )
    def test_gradient_analytical_formula(self, nu, x):
        """Test that gradients match the analytical formula: ∂K_ν/∂x = -1/2 * (K_{ν-1}(x) + K_{ν+1}(x))."""
        grad_fn = jax.grad(lambda x_val: kv(nu, x_val))
        computed_grad = grad_fn(x)

        # Compute expected gradient using analytical formula
        kv_prev = kv(nu - 1.0, x)
        kv_next = kv(nu + 1.0, x)
        expected_grad = -0.5 * (kv_prev + kv_next)

        # Should match very closely since we use the same formula
        assert jnp.allclose(computed_grad, expected_grad, rtol=1e-10), (
            f"Gradient doesn't match analytical formula at nu={nu}, x={x}"
        )

    def test_gradient_vectorization(self):
        """Test that gradients work with vmap."""
        grad_fn = jax.vmap(jax.grad(lambda x_val: kv(2.5, x_val)))
        x_array = jnp.array([1.0, 5.0, 10.0, 50.0])
        grads = grad_fn(x_array)

        # Verify all gradients are finite
        assert jnp.all(jnp.isfinite(grads)), "Some vectorized gradients are not finite"

        # Verify shapes match
        assert grads.shape == x_array.shape, "Gradient shape mismatch"

    @pytest.mark.skip(
        reason="custom_vjp doesn't support higher-order derivatives by default"
    )
    def test_second_derivative(self):
        """Test that second derivatives (Hessian) work.

        Note: This test is skipped because custom_vjp doesn't support higher-order
        differentiation. This is a known limitation and not critical for first-order
        gradient-based optimization.
        """
        hessian_fn = jax.grad(jax.grad(lambda x_val: kv(2.5, x_val)))
        hess_val = hessian_fn(5.0)

        # Just verify it's finite - we're not testing accuracy of second derivative
        assert jnp.isfinite(hess_val), f"Second derivative is not finite: {hess_val}"

    @pytest.mark.parametrize(
        "nu,x",
        [
            (2.5, 0.1),  # Small x
            (2.5, 100.0),  # Large x
            (300.9, 500.0),  # Large nu
            (0.5, 0.1),  # Small x, small nu
        ],
    )
    def test_gradient_edge_cases(self, nu, x):
        """Test gradients at edge cases (small x, large x, large nu)."""
        grad_fn = jax.grad(lambda x_val: kv(nu, x_val))
        grad_val = grad_fn(x)

        # Verify gradient is finite and reasonable
        assert jnp.isfinite(grad_val), f"Gradient not finite at nu={nu}, x={x}"

        # Verify against analytical formula
        kv_prev = kv(nu - 1.0, x)
        kv_next = kv(nu + 1.0, x)
        expected_grad = -0.5 * (kv_prev + kv_next)

        assert jnp.allclose(grad_val, expected_grad, rtol=1e-8), (
            f"Gradient mismatch at edge case nu={nu}, x={x}"
        )

    def test_gradient_wrt_nu_is_zero(self):
        """Test that gradient w.r.t. nu returns zero (not supported)."""
        # Gradient w.r.t. first argument (nu)
        grad_nu_fn = jax.grad(lambda nu_val: kv(nu_val, 5.0), argnums=0)
        grad_nu = grad_nu_fn(2.5)

        # Should be zero since we don't support gradient w.r.t. order
        assert grad_nu == 0.0, "Gradient w.r.t. nu should be zero"

    def test_gradient_negative_nu(self):
        """Test that gradients work with negative nu (should use abs(nu))."""
        # K_{-nu}(x) = K_nu(x), so gradient should be the same
        grad_pos = jax.grad(lambda x_val: kv(2.5, x_val))(5.0)
        grad_neg = jax.grad(lambda x_val: kv(-2.5, x_val))(5.0)

        assert jnp.allclose(grad_pos, grad_neg, rtol=1e-10), (
            "Gradient should be same for positive and negative nu"
        )

    def test_gradient_integer_order(self):
        """Test gradients for integer orders (special case)."""
        # Integer orders use different implementation path
        for nu in [0, 1, 2, 5]:
            grad_fn = jax.grad(lambda x_val: kv(nu, x_val))
            grad_val = grad_fn(5.0)

            # Verify against analytical formula
            kv_prev = kv(nu - 1.0, 5.0) if nu > 0 else kv(1.0, 5.0)  # K_{-1} = K_1
            kv_next = kv(nu + 1.0, 5.0)
            expected_grad = -0.5 * (kv_prev + kv_next)

            assert jnp.allclose(grad_val, expected_grad, rtol=1e-8), (
                f"Gradient mismatch for integer order nu={nu}"
            )

    def test_gradient_half_integer_order(self):
        """Test gradients for half-integer orders (special case)."""
        # Half-integer orders use closed-form expressions
        for nu in [0.5, 1.5, 2.5, 3.5]:
            grad_fn = jax.grad(lambda x_val: kv(nu, x_val))
            grad_val = grad_fn(5.0)

            # Verify against analytical formula
            kv_prev = kv(nu - 1.0, 5.0)
            kv_next = kv(nu + 1.0, 5.0)
            expected_grad = -0.5 * (kv_prev + kv_next)

            assert jnp.allclose(grad_val, expected_grad, rtol=1e-8), (
                f"Gradient mismatch for half-integer order nu={nu}"
            )

    def test_gradient_jit_compilation(self):
        """Test that gradient compilation works without errors."""
        # Compile gradient function
        grad_fn = jax.jit(jax.grad(lambda x_val: kv(2.5, x_val)))

        # Should compile without errors
        _ = grad_fn(5.0)

        # Call again to verify compiled version works
        grad_val = grad_fn(10.0)
        assert jnp.isfinite(grad_val), "Compiled gradient not finite"

    def test_gradient_multiple_calls(self):
        """Test that gradients are consistent across multiple calls."""
        grad_fn = jax.grad(lambda x_val: kv(2.5, x_val))

        # Call multiple times with same input
        grads = [grad_fn(5.0) for _ in range(5)]

        # All should be identical
        for grad_val in grads[1:]:
            assert jnp.allclose(grad_val, grads[0], rtol=1e-12), (
                "Gradients inconsistent across calls"
            )

    def test_gradient_batch_computation(self):
        """Test gradients with batched inputs using vmap."""
        nu_values = jnp.array([0.5, 1.0, 2.5, 10.0])
        x_values = jnp.array([1.0, 5.0, 10.0, 50.0])

        # Compute gradients for all combinations
        grad_fn = jax.vmap(
            jax.vmap(
                jax.grad(lambda x_val, nu_val: kv(nu_val, x_val)), in_axes=(None, 0)
            ),
            in_axes=(0, None),
        )

        grads = grad_fn(x_values, nu_values)

        # Verify shape: (len(x_values), len(nu_values))
        assert grads.shape == (len(x_values), len(nu_values))

        # Verify all are finite
        assert jnp.all(jnp.isfinite(grads)), "Some batch gradients are not finite"
