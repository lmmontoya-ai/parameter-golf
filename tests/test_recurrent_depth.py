from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = f"torch is required for Recurrent Depth tests: {exc}"
else:
    IMPORT_ERROR = ""

if torch is not None:
    from research.architectures.recurrent_depth import RecurrentDepthConfig
    from train_gpt import RecurrentGPT, split_named_optimizer_params


def make_recurrent_model(
    *,
    num_layers: int = 9,
    prelude_layers: int = 1,
    core_layers: int = 2,
    steps: int = 3,
    backprop_steps: int = 3,
    coda_layers: int = 2,
    eval_steps: int = 3,
) -> RecurrentGPT:
    return RecurrentGPT(
        vocab_size=32,
        num_layers=num_layers,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        recurrent_config=RecurrentDepthConfig(
            enabled=True,
            prelude_layers=prelude_layers,
            core_layers=core_layers,
            steps=steps,
            backprop_steps=backprop_steps,
            coda_layers=coda_layers,
            eval_steps=eval_steps,
            state_init="like_init",
            input_injection="linear_concat",
        ),
    )


@unittest.skipIf(torch is None, IMPORT_ERROR)
class RecurrentDepthIntegrationTest(unittest.TestCase):
    def test_forward_and_backward_with_fixed_recurrence(self):
        model = make_recurrent_model()
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        loss = model(input_ids, target_ids)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsNotNone(model.adapter.weight.grad)
        self.assertEqual(model.adapter.weight.grad.shape, model.adapter.weight.shape)

    def test_forward_and_backward_with_truncated_prefix(self):
        model = make_recurrent_model(backprop_steps=1)
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        loss = model(input_ids, target_ids)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsNotNone(model.core_blocks[0].attn.c_q.weight.grad)

    def test_effective_depth_validation_rejects_invalid_schedule(self):
        with self.assertRaisesRegex(ValueError, "must equal NUM_LAYERS"):
            _ = make_recurrent_model(
                num_layers=9,
                prelude_layers=1,
                core_layers=2,
                steps=2,
                backprop_steps=2,
                coda_layers=2,
            )

    def test_shared_core_blocks_are_reused_across_steps(self):
        model = make_recurrent_model()
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))
        counts = [0 for _ in model.core_blocks]
        hooks = []

        for idx, block in enumerate(model.core_blocks):
            hooks.append(
                block.register_forward_hook(
                    lambda _module, _inputs, _output, idx=idx, counts=counts: counts.__setitem__(idx, counts[idx] + 1)
                )
            )

        try:
            _ = model(input_ids, target_ids)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(counts, [3, 3])

    def test_skip_weights_are_absent_in_recurrent_model(self):
        model = make_recurrent_model()
        self.assertFalse(hasattr(model, "skip_weights"))

    def test_adapter_matrix_stays_in_matrix_group(self):
        model = make_recurrent_model()
        matrix_params, scalar_params = split_named_optimizer_params(model._iter_named_block_params())
        matrix_param_ids = {id(param) for param in matrix_params}
        scalar_param_ids = {id(param) for param in scalar_params}

        self.assertIn(id(model.adapter.weight), matrix_param_ids)
        self.assertNotIn(id(model.adapter.weight), scalar_param_ids)

    def test_control_params_stay_in_scalar_group(self):
        model = make_recurrent_model()
        matrix_params, scalar_params = split_named_optimizer_params(model._iter_named_block_params())
        matrix_param_ids = {id(param) for param in matrix_params}
        scalar_param_ids = {id(param) for param in scalar_params}

        for block in list(model.prelude_blocks) + list(model.core_blocks) + list(model.coda_blocks):
            self.assertIn(id(block.attn_scale), scalar_param_ids)
            self.assertIn(id(block.mlp_scale), scalar_param_ids)
            self.assertNotIn(id(block.attn_scale), matrix_param_ids)
            self.assertNotIn(id(block.mlp_scale), matrix_param_ids)


if __name__ == "__main__":
    unittest.main()
