"""The single-pass trunk API composes to `get_pairformer_output`.

Builds the released checkpoint (downloaded on first use; needs the `reference`
dependency group)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jopendde.inference import Predictor, spec_from_sequences

N_CYCLE = 3


@pytest.fixture(scope="module")
def model_and_features():
    predictor = Predictor.from_checkpoint()
    inputs = predictor.featurize(spec_from_sequences(["MKTAYIAKQRQISFVKSHFSRQ"], name="t", seed=0))
    return predictor.model, inputs.feat


def test_steps_compose_to_get_pairformer_output(model_and_features):
    model, feat = model_and_features

    @eqx.filter_jit
    def by_steps(model, feat):
        inputs = model.embed(feat)
        state = model.init_state(inputs)
        for _ in range(N_CYCLE):
            state = model.step(feat, inputs, state)
        return inputs.s_inputs, state.s, state.z

    expected = eqx.filter_jit(lambda m, f: m.get_pairformer_output(f, N_CYCLE))(model, feat)
    for a, b in zip(by_steps(model, feat), expected):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_get_pairformer_output_differentiates_the_last_pass_only(model_and_features):
    model, feat = model_and_features

    def scaled(feat, scale):
        return eqx.tree_at(lambda f: f.profile, feat, feat.profile * scale)

    @eqx.filter_jit
    def by_loop(model, feat, scale):
        return jnp.sum(model.get_pairformer_output(scaled(feat, scale), N_CYCLE)[2] ** 2)

    @eqx.filter_jit
    def by_detached_steps(model, feat, scale):
        f = scaled(feat, scale)
        inputs = model.embed(f)
        state = model.init_state(inputs)
        for _ in range(N_CYCLE):
            state = model.step(f, inputs, jax.lax.stop_gradient(state))
        return jnp.sum(state.z**2)

    scale = jnp.float32(1.0)
    g = jax.grad(by_loop, argnums=2)(model, feat, scale)
    g_manual = jax.grad(by_detached_steps, argnums=2)(model, feat, scale)
    assert g != 0
    np.testing.assert_allclose(g, g_manual, rtol=1e-4)
