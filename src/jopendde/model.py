"""Top-level OpenDDE model: wires the submodules into the full inference
forward pass.

Pipeline: input embedding -> recycling trunk (template + MSA + Pairformer) ->
structural-token expansion (+ refiner) -> diffusion sampling -> distogram/
contact + confidence heads. FoldCP paths, memory-optimization args,
N_model_seed > 1, shape-complementarity, and the numpy confidence
post-processing are not implemented.

With `pair_output_space == "residue"` the distogram and confidence heads
consume the residue branch. The sampled `coordinate` is stochastic (Algorithm
18's Euler sampler draws fresh Gaussian noise each step).

The recycling trunk can also be driven one pass at a time; `get_pairformer_output`
is built on this:

    inputs = model.embed(feat)                 # fixed across passes
    state = model.init_state(inputs)           # zeros
    state = model.step(feat, inputs, state)    # one pass (repeat)
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jopendde.backend import Linear, LayerNorm
from jopendde.confidence import ConfidenceHead, PairformerStack
from jopendde.diffusion import DiffusionModule, noise_schedule, sample_diffusion
from jopendde.embedders import InputFeatureEmbedder, RelativePositionEncoding
from jopendde.features import Features, Prediction
from jopendde.head import DistogramHead
from jopendde.msa import MSAModule, TemplateEmbedder
from jopendde.structural_tokens import StructuralTokenExpander


class TrunkInputs(eqx.Module):
    """What every trunk pass reads that stays fixed across passes (AF3 Algorithm 1
    lines 1-5): the per-token input features and the initial single / pair."""

    s_inputs: Float[Array, "N Cs_in"]
    s_init: Float[Array, "N Cs"]
    z_init: Float[Array, "N N Cz"]


class TrunkState(eqx.Module):
    """The recycled residue-branch single and pair representations."""

    s: Float[Array, "N Cs"]
    z: Float[Array, "N N Cz"]


class OpenDDE(eqx.Module):
    # --- submodules ---
    input_embedder: InputFeatureEmbedder
    relative_position_encoding: RelativePositionEncoding
    template_embedder: TemplateEmbedder
    msa_module: MSAModule
    pairformer_stack: PairformerStack
    diffusion_module: DiffusionModule
    distogram_head: DistogramHead
    confidence_head: ConfidenceHead
    structural_token_expander: StructuralTokenExpander
    structural_token_refiner: PairformerStack

    # --- top-level trainable params ---
    linear_no_bias_sinit: Linear
    linear_no_bias_zinit1: Linear
    linear_no_bias_zinit2: Linear
    linear_no_bias_token_bond: Linear
    linear_no_bias_z_cycle: Linear
    linear_no_bias_s: Linear
    layernorm_z_cycle: LayerNorm
    layernorm_s: LayerNorm

    # --- static config ---
    # Several architectural switches are FIXED by the opendde_v1 checkpoint and
    # are deliberately NOT stored as fields -- their branches are hardcoded and
    # the assumptions are asserted in from_torch(). Fixed to:
    #   enable_structural_token_expansion = True   (always expand)
    #   enable_structural_token_refiner   = True   (always run the refiner)
    #   template_embedder.n_blocks        > 0      (always embed templates)
    #   pair_output_space                 = "residue"  (heads read residue branch)
    # A checkpoint that violated any of these would silently take a dead path,
    # so from_torch() asserts them instead.
    #
    # The per-run compute knobs -- N_cycle (recycles), N_sample, N_step -- are
    # NOT fields: they're inference-time choices, not learned/architectural
    # properties, so they're passed as arguments to predict()/get_pairformer_
    # output()/sample_coordinates() instead of baked in (a converted model then
    # serves any sampling budget without a rebuild). Only checkpoint-intrinsic
    # config lives below.

    # distogram / contact: the distogram head was trained to emit `dist_no_bins`
    # logits over [dist_min_bin, dist_max_bin]; fixed by the weights, not a knob.
    dist_min_bin: float
    dist_max_bin: float
    dist_no_bins: int

    # diffusion sampler dynamics + noise schedule shape (fixed inference
    # hyperparameters for opendde_v1; sched_sigma_data is the training
    # data-normalization constant, == diffusion_module.sigma_data).
    gamma0: float
    gamma_min: float
    noise_scale_lambda: float
    step_scale_eta: float
    sched_s_max: float
    sched_s_min: float
    sched_rho: float
    sched_sigma_data: float

    # ---------------------------------------------------------------------
    # construction
    # ---------------------------------------------------------------------
    @classmethod
    def from_torch(cls, model):
        from jopendde.backend import from_torch

        sd = model.configs.sample_diffusion
        ns = dict(model.configs.inference_noise_scheduler)
        dist = model.configs.confidence.distogram

        # Architectural switches hardcoded here (see the field-block note).
        # Assert the torch config matches so a mismatched checkpoint fails loudly
        # at load rather than silently exercising a branch that isn't there.
        assert bool(model.enable_structural_token_expansion), (
            "jopendde hardcodes structural-token expansion ON, but the torch "
            "model has enable_structural_token_expansion=False"
        )
        assert bool(model.enable_structural_token_refiner), (
            "jopendde hardcodes the structural-token refiner ON, but the torch "
            "model has enable_structural_token_refiner=False"
        )
        assert str(model.pair_output_space) == "residue", (
            "jopendde hardcodes pair_output_space='residue', but the torch model "
            f"has pair_output_space={model.pair_output_space!r}"
        )
        assert int(model.template_embedder.n_blocks) > 0, (
            "jopendde hardcodes template embedding ON, but the torch model has "
            "template_embedder.n_blocks == 0"
        )
        assert model.structural_token_expander is not None
        assert model.structural_token_refiner is not None

        return cls(
            input_embedder=from_torch(model.input_embedder),
            relative_position_encoding=from_torch(model.relative_position_encoding),
            template_embedder=from_torch(model.template_embedder),
            msa_module=from_torch(model.msa_module),
            pairformer_stack=from_torch(model.pairformer_stack),
            diffusion_module=from_torch(model.diffusion_module),
            distogram_head=from_torch(model.distogram_head),
            confidence_head=from_torch(model.confidence_head),
            structural_token_expander=from_torch(model.structural_token_expander),
            structural_token_refiner=from_torch(model.structural_token_refiner),
            linear_no_bias_sinit=from_torch(model.linear_no_bias_sinit),
            linear_no_bias_zinit1=from_torch(model.linear_no_bias_zinit1),
            linear_no_bias_zinit2=from_torch(model.linear_no_bias_zinit2),
            linear_no_bias_token_bond=from_torch(model.linear_no_bias_token_bond),
            linear_no_bias_z_cycle=from_torch(model.linear_no_bias_z_cycle),
            linear_no_bias_s=from_torch(model.linear_no_bias_s),
            layernorm_z_cycle=from_torch(model.layernorm_z_cycle),
            layernorm_s=from_torch(model.layernorm_s),
            dist_min_bin=float(dist.min_bin),
            dist_max_bin=float(dist.max_bin),
            dist_no_bins=int(dist.no_bins),
            gamma0=float(sd.get("gamma0")),
            gamma_min=float(sd.get("gamma_min")),
            noise_scale_lambda=float(sd.get("noise_scale_lambda")),
            step_scale_eta=float(sd.get("step_scale_eta")),
            sched_s_max=float(ns["s_max"]),
            sched_s_min=float(ns["s_min"]),
            sched_rho=float(ns["rho"]),
            sched_sigma_data=float(ns["sigma_data"]),
        )

    # ---------------------------------------------------------------------
    # trunk
    # ---------------------------------------------------------------------
    def generate_relp(self, feat: Features) -> Float[Array, "N N F"]:
        return self.relative_position_encoding.generate_relp(
            feat.asym_id,
            feat.residue_index,
            feat.entity_id,
            feat.token_index,
            feat.sym_id,
        )

    def embed(self, feat: Features) -> TrunkInputs:
        """The trunk inputs that stay fixed across passes."""
        s_inputs = self.input_embedder(feat)  # [N_token, 449]
        s_init = self.linear_no_bias_sinit(s_inputs)  # [N_token, c_s]

        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        z_init = z_init + self.relative_position_encoding(feat.relp)
        z_init = z_init + self.linear_no_bias_token_bond(feat.token_bonds[..., None])
        return TrunkInputs(s_inputs=s_inputs, s_init=s_init, z_init=z_init)

    def init_state(self, inputs: TrunkInputs) -> TrunkState:
        """The recycled state before the first pass (zeros)."""
        return TrunkState(s=jnp.zeros_like(inputs.s_init), z=jnp.zeros_like(inputs.z_init))

    def step(self, feat: Features, inputs: TrunkInputs, state: TrunkState) -> TrunkState:
        """Exactly one trunk pass (AF3 Algorithm 1 lines 8-13). Differentiates
        through `state`; `get_pairformer_output` detaches it."""
        z = inputs.z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(state.z))
        z = z + self.template_embedder(feat, z)  # templates always embedded
        z = self.msa_module(feat, z, inputs.s_inputs, pair_mask=None)
        s = inputs.s_init + self.linear_no_bias_s(self.layernorm_s(state.s))
        s, z = self.pairformer_stack(s, z, pair_mask=None)
        return TrunkState(s=s, z=z)

    def get_pairformer_output(self, feat: Features, n_cycle: int):
        """Returns the residue branch (s_inputs, s, z). `n_cycle` (recycles) is
        a static inference knob passed by the caller.

        Equivalent to `embed`, `init_state`, then `n_cycle` calls to `step`, each
        on a detached state: recycling is a fixed-point-ish refinement, not a path
        to backprop through, so the gradient is confined to the final pass,
        bounding backward memory."""
        inputs = self.embed(feat)

        def cycle(state, _):
            return self.step(feat, inputs, jax.lax.stop_gradient(state)), None

        state, _ = jax.lax.scan(cycle, self.init_state(inputs), None, length=n_cycle)
        return inputs.s_inputs, state.s, state.z

    # ---------------------------------------------------------------------
    # structural-token expansion
    # ---------------------------------------------------------------------
    def expand_to_structural_tokens(self, feat: Features, s_inputs, s, z):
        """Expansion + refiner are always on for opendde_v1 (asserted in
        from_torch). Returns the structural branch (struct_feat, s_inputs, s, z)
        plus `attn_bias`, the structural-token attention bias -- a computed
        intermediate passed explicitly to the refiner + diffusion, not stored on
        `Features`."""
        parent = feat.parent_residue_idx.astype(jnp.int32)
        s_inputs_st, s_st, z_st, attn_bias = self.structural_token_expander(
            feat, s_inputs, s, z
        )

        # Rebuild the feature view for the structural-token branch: the same
        # metadata re-pointed at the expanded token set (typed field swaps via
        # dataclasses.replace).
        take = lambda x: jnp.take(x, parent, axis=-1)
        struct_feat = dataclasses.replace(
            feat,
            token_index=feat.structural_token_index,
            atom_to_token_idx=feat.atom_to_structural_token_idx,
            atom_to_tokatom_idx=feat.atom_to_structural_tokatom_idx,
            asym_id=take(feat.asym_id),
            residue_index=take(feat.residue_index),
            entity_id=take(feat.entity_id),
            sym_id=take(feat.sym_id),
            has_frame=feat.structural_has_frame,
            frame_atom_index=feat.structural_frame_atom_index,
            pae_rep_atom_mask=feat.structural_pae_rep_atom_mask,
            distogram_rep_atom_mask=feat.structural_distogram_rep_atom_mask,
        )
        struct_feat = dataclasses.replace(struct_feat, relp=self.generate_relp(struct_feat))

        s_st, z_st = self.structural_token_refiner(
            s_st, z_st, pair_mask=None, extra_attn_bias=attn_bias
        )
        return struct_feat, s_inputs_st, s_st, z_st, attn_bias

    # ---------------------------------------------------------------------
    # heads
    # ---------------------------------------------------------------------
    def contact_probs(
        self, z_pair: Float[Array, "N N Cz"], thres: float = 8.0
    ) -> Float[Array, "N N"]:
        """distogram_head -> softmax -> sum of bins whose top edge <= thres."""
        logits = self.distogram_head(z_pair)
        probs = jax.nn.softmax(logits, axis=-1)
        breaks = jnp.linspace(self.dist_min_bin, self.dist_max_bin, self.dist_no_bins - 1)
        bin_tops = jnp.concatenate([breaks, jnp.array([jnp.inf], dtype=breaks.dtype)])
        return jnp.sum(probs * (bin_tops <= thres), axis=-1)

    def run_confidence_head(self, feat, s_inputs, s_trunk, z_trunk, x_pred_coords):
        return self.confidence_head(
            feat,
            s_inputs,
            s_trunk,
            z_trunk,
            pair_mask=None,
            x_pred_coords=x_pred_coords,
        )

    # ---------------------------------------------------------------------
    # diffusion sampling
    # ---------------------------------------------------------------------
    def sample_coordinates(self, struct_feat, s_inputs, s_trunk, z_trunk, key, *, n_sample: int, n_step: int, extra_attn_bias):
        schedule = noise_schedule(
            n_step,
            s_max=self.sched_s_max,
            s_min=self.sched_s_min,
            rho=self.sched_rho,
            sigma_data=self.sched_sigma_data,
        )
        return sample_diffusion(
            self.diffusion_module,
            struct_feat,
            s_inputs,
            s_trunk,
            z_trunk,
            schedule,
            key,
            N_sample=n_sample,
            gamma0=self.gamma0,
            gamma_min=self.gamma_min,
            noise_scale_lambda=self.noise_scale_lambda,
            step_scale_eta=self.step_scale_eta,
            extra_attn_bias=extra_attn_bias,
        )

    # ---------------------------------------------------------------------
    # full forward
    # ---------------------------------------------------------------------
    def predict(self, feat: Features, key, *, n_cycle: int, n_sample: int, n_step: int) -> Prediction:
        s_inputs, s, z = self.get_pairformer_output(feat, n_cycle)

        # Diffusion runs on the structural-token branch; the confidence and
        # distogram heads read the residue branch (pair_output_space="residue",
        # asserted at load).
        struct_feat, s_inputs_st, s_st, z_st, attn_bias = self.expand_to_structural_tokens(
            feat, s_inputs, s, z
        )

        coordinate = self.sample_coordinates(
            struct_feat, s_inputs_st, s_st, z_st, key,
            n_sample=n_sample, n_step=n_step, extra_attn_bias=attn_bias,
        )
        contact = self.contact_probs(z)
        plddt, pae, pde, resolved = self.run_confidence_head(
            feat, s_inputs, s, z, coordinate
        )
        return Prediction(
            coordinate=coordinate,
            contact_probs=contact,
            plddt=plddt,
            pae=pae,
            pde=pde,
            resolved=resolved,
        )
