from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = f"torch is required for Attention Residuals tests: {exc}"
else:
    IMPORT_ERROR = ""

if torch is not None:
    from research.architectures.attention_residuals import BlockAttentionResidual
    from train_gpt import GPT, split_block_optimizer_params


def make_test_model(num_layers: int, attnres_enable: bool, attnres_block_layers: int) -> GPT:
    return GPT(
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
        attnres_enable=attnres_enable,
        attnres_block_layers=attnres_block_layers,
    )


@unittest.skipIf(torch is None, IMPORT_ERROR)
class BlockAttentionResidualTest(unittest.TestCase):
    def test_shape_and_dtype_match_partial_block(self):
        module = BlockAttentionResidual(model_dim=8)
        completed_blocks = [torch.randn(2, 3, 8), torch.randn(2, 3, 8)]
        partial_block = torch.randn(2, 3, 8)

        out = module(completed_blocks, partial_block)

        self.assertEqual(out.shape, partial_block.shape)
        self.assertEqual(out.dtype, partial_block.dtype)

    def test_zero_query_returns_identical_candidates(self):
        module = BlockAttentionResidual(model_dim=8)
        candidate = torch.randn(2, 3, 8)

        out = module([candidate.clone(), candidate.clone()], candidate.clone())

        torch.testing.assert_close(out, candidate, atol=1e-6, rtol=1e-6)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class AttentionResidualIntegrationTest(unittest.TestCase):
    def test_disabled_path_forward_and_backward(self):
        model = make_test_model(num_layers=4, attnres_enable=False, attnres_block_layers=3)
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        loss = model(input_ids, target_ids)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        grad = model.tok_emb.weight.grad
        self.assertIsNotNone(grad)
        self.assertEqual(grad.shape, model.tok_emb.weight.shape)

    def test_enabled_path_forward_and_backward(self):
        model = make_test_model(num_layers=4, attnres_enable=True, attnres_block_layers=3)
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        loss = model(input_ids, target_ids)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsNotNone(model.blocks[0].attn_res_attn)
        self.assertIsNotNone(model.blocks[0].attn_res_attn.query.grad)

    def test_attnres_query_params_stay_in_scalar_group(self):
        model = make_test_model(num_layers=4, attnres_enable=True, attnres_block_layers=2)
        matrix_params, scalar_params = split_block_optimizer_params(model.blocks)
        matrix_param_ids = {id(param) for param in matrix_params}
        scalar_param_ids = {id(param) for param in scalar_params}

        for block in model.blocks:
            for module in (block.attn_res_attn, block.attn_res_mlp):
                self.assertIsNotNone(module)
                assert module is not None
                self.assertEqual(module.query.ndim, 1)
                self.assertIn(id(module.query), scalar_param_ids)
                self.assertNotIn(id(module.query), matrix_param_ids)

    def test_block_boundary_counts_progress_by_block(self):
        model = make_test_model(num_layers=9, attnres_enable=True, attnres_block_layers=3)
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))
        counts: list[int] = []
        hooks = []

        for block in model.blocks:
            assert block.attn_res_attn is not None
            hooks.append(
                block.attn_res_attn.register_forward_pre_hook(
                    lambda _module, inputs, counts=counts: counts.append(len(inputs[0]))
                )
            )

        try:
            _ = model(input_ids, target_ids)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(counts, [1, 1, 1, 2, 2, 2, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
