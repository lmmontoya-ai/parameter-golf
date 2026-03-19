from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = f"torch is required for dense screening tests: {exc}"
else:
    IMPORT_ERROR = ""

if torch is not None:
    import train_gpt as tg


def make_dense_model(
    *,
    low_rank_ffn_enable: bool = False,
    low_rank_ffn_ratio: float = 0.5,
    mtp_enable: bool = False,
    mtp_k: int = 2,
    mtp_loss_weight: float = 0.3,
) -> tg.GPT:
    return tg.GPT(
        vocab_size=32,
        num_layers=4,
        model_dim=32,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        attnres_enable=False,
        attnres_block_layers=3,
        low_rank_ffn_enable=low_rank_ffn_enable,
        low_rank_ffn_ratio=low_rank_ffn_ratio,
        mtp_enable=mtp_enable,
        mtp_k=mtp_k,
        mtp_loss_weight=mtp_loss_weight,
    )


@unittest.skipIf(torch is None, IMPORT_ERROR)
class LowRankFfnTest(unittest.TestCase):
    def test_low_rank_variant_has_fewer_parameters(self):
        dense = make_dense_model()
        low_rank = make_dense_model(low_rank_ffn_enable=True, low_rank_ffn_ratio=0.5)

        dense_params = sum(p.numel() for p in dense.parameters())
        low_rank_params = sum(p.numel() for p in low_rank.parameters())

        self.assertLess(low_rank_params, dense_params)

    def test_low_rank_variant_forward_and_backward(self):
        model = make_dense_model(low_rank_ffn_enable=True, low_rank_ffn_ratio=0.5)
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        loss = model(input_ids, target_ids)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsNotNone(model.blocks[0].mlp.fc.reduce.weight.grad)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class MultiTokenPredictionTest(unittest.TestCase):
    def test_mtp_variant_forward_and_backward(self):
        model = make_dense_model(mtp_enable=True, mtp_k=2, mtp_loss_weight=0.3)
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        loss = model(input_ids, target_ids)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertIsNotNone(model.tok_emb.weight.grad)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class QuantizationPassThroughTest(unittest.TestCase):
    def test_large_tensor_fp16_passthrough_allowlist(self):
        old_patterns = tg.INT8_PASSTHROUGH_FP16_NAME_PATTERNS
        try:
            tg.INT8_PASSTHROUGH_FP16_NAME_PATTERNS = ("big.weight",)
            state_dict = {"big.weight": torch.randn(1024, 128)}
            quant_obj, _ = tg.quantize_state_dict_int8(state_dict)
            self.assertIn("big.weight", quant_obj["passthrough"])
            self.assertEqual(quant_obj["passthrough"]["big.weight"].dtype, torch.float16)
            restored = tg.dequantize_state_dict_int8(quant_obj)
            self.assertEqual(restored["big.weight"].dtype, torch.float32)
        finally:
            tg.INT8_PASSTHROUGH_FP16_NAME_PATTERNS = old_patterns


if __name__ == "__main__":
    unittest.main()
