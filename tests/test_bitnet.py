from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = f"torch is required for BitNet tests: {exc}"
else:
    IMPORT_ERROR = ""

if torch is not None:
    import train_gpt as tg


@unittest.skipIf(torch is None, IMPORT_ERROR)
class TernaryPackingTest(unittest.TestCase):
    def test_pack_unpack_roundtrip(self):
        codes = torch.tensor([[-1, 0, 1, -1, 1], [0, 1, 0, -1, 1]], dtype=torch.int8)
        packed = tg.pack_ternary_codes(codes)
        unpacked = tg.unpack_ternary_codes(packed, shape=tuple(codes.shape))
        self.assertTrue(torch.equal(unpacked, codes))

    def test_quantize_ternary_tensor_scale_modes(self):
        weight = torch.randn(6, 8)
        q_row, s_row = tg.quantize_ternary_tensor(weight, scale_mode="per_row_absmean")
        q_tensor, s_tensor = tg.quantize_ternary_tensor(weight, scale_mode="per_tensor_absmean")
        self.assertEqual(q_row.shape, weight.shape)
        self.assertEqual(q_tensor.shape, weight.shape)
        self.assertEqual(s_row.shape, torch.Size([weight.shape[0]]))
        self.assertEqual(s_tensor.shape, torch.Size([]))
        self.assertLessEqual(int(q_row.abs().max().item()), 1)
        self.assertLessEqual(int(q_tensor.abs().max().item()), 1)

    def test_fake_ternary_weight_ste_backward(self):
        weight = torch.randn(4, 8, requires_grad=True)
        out = tg.fake_ternary_weight_ste(weight, scale_mode="per_tensor_absmean").sum()
        out.backward()
        self.assertIsNotNone(weight.grad)
        self.assertEqual(weight.grad.shape, weight.shape)

    def test_hadamard_transform_preserves_shape_and_norm(self):
        x = torch.randn(3, 4, 8)
        y = tg.hadamard_transform_last_dim(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(x.norm(), y.norm(), rtol=1e-4, atol=1e-4))


@unittest.skipIf(torch is None, IMPORT_ERROR)
class TernaryExportTest(unittest.TestCase):
    def test_ternary_rowpack_quantizes_eligible_dense_weights(self):
        state_dict = {
            "blocks.0.attn.c_q.weight": torch.randn(384, 256),
            "blocks.0.attn.c_q.bias": torch.randn(384),
            "blocks.0.attn.q_gain": torch.ones(2),
        }
        ternary_config = tg.TernaryExportConfig(
            enabled=True,
            track="artifact",
            scale_mode="per_row_absmean",
            scope="transformer_matrices",
        )
        quant_obj, _ = tg.quantize_state_dict(
            state_dict,
            export_format="ternary_rowpack_zlib",
            passthrough_fp16_patterns=(),
            ternary_config=ternary_config,
        )
        self.assertEqual(quant_obj["qmeta"]["blocks.0.attn.c_q.weight"]["scheme"], "ternary_packed")
        restored = tg.dequantize_state_dict(quant_obj)
        self.assertEqual(restored["blocks.0.attn.c_q.weight"].shape, state_dict["blocks.0.attn.c_q.weight"].shape)
        self.assertEqual(restored["blocks.0.attn.c_q.bias"].shape, state_dict["blocks.0.attn.c_q.bias"].shape)

    def test_resolve_final_eval_export_formats_includes_control_and_ternary(self):
        class Args:
            export_format = "ternary_rowpack_zlib"
            final_eval_export_formats = ""

        self.assertEqual(
            tg.resolve_final_eval_export_formats(Args(), ternary_enabled=True),
            ("ternary_rowpack_zlib", "int8_zlib"),
        )


@unittest.skipIf(torch is None, IMPORT_ERROR)
class ModelBitNetTest(unittest.TestCase):
    def test_configure_model_ternary_qat_marks_only_eligible_dense_modules(self):
        model = tg.GPT(
            vocab_size=64,
            num_layers=2,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.02,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            attnres_enable=False,
            attnres_block_layers=1,
        )
        tg.assign_linear_module_names(model)
        tg.configure_model_ternary_qat(
            model,
            enabled=True,
            scale_mode="per_row_absmean",
            track="artifact",
        )
        enabled_modules = {
            name
            for name, module in model.named_modules()
            if isinstance(module, tg.CastedLinear) and module.ternary_qat_enabled
        }
        self.assertIn("blocks.0.attn.c_q", enabled_modules)
        self.assertIn("blocks.0.attn.c_k", enabled_modules)
        self.assertIn("blocks.0.attn.c_v", enabled_modules)
        self.assertIn("blocks.0.attn.proj", enabled_modules)
        self.assertIn("blocks.0.mlp.fc", enabled_modules)
        self.assertIn("blocks.0.mlp.proj", enabled_modules)

    def test_native_bitnet_forward_backward(self):
        model = tg.BitNetGPT(
            vocab_size=64,
            num_layers=2,
            model_dim=16,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.02,
            logit_softcap=30.0,
            rope_base=10000.0,
            act_bits=8,
            hadamard_enable=False,
            hadamard_scope=frozenset(),
        )
        x = torch.randint(0, 64, (2, 8), dtype=torch.int64)
        y = torch.randint(0, 64, (2, 8), dtype=torch.int64)
        loss = model(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.blocks[0].attn.wq.weight.grad)

    def test_native_lowbit_runtime_attaches_to_bitlinear_modules(self):
        model = tg.BitNetGPT(
            vocab_size=64,
            num_layers=1,
            model_dim=512,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=2,
            tie_embeddings=True,
            tied_embed_init_std=0.02,
            logit_softcap=30.0,
            rope_base=10000.0,
            act_bits=8,
            hadamard_enable=True,
            hadamard_scope=frozenset({"wo", "wdown"}),
        )
        tg.assign_linear_module_names(model)
        ternary_config = tg.TernaryExportConfig(
            enabled=True,
            track="native",
            scale_mode="per_tensor_absmean",
            scope="transformer_matrices",
        )
        quant_obj, _ = tg.quantize_state_dict(
            model.state_dict(),
            export_format="ternary_rowpack_zlib",
            passthrough_fp16_patterns=(),
            ternary_config=ternary_config,
        )
        tg.configure_model_lowbit_runtime(model, quant_obj, backend="reference")
        self.assertIsNotNone(model.blocks[0].attn.wq.lowbit_runtime_state)
        self.assertEqual(model.blocks[0].attn.wq.lowbit_runtime_state.activation_bits, 8)
        self.assertTrue(model.blocks[0].attn.wo.lowbit_runtime_state.hadamard)
        self.assertTrue(model.blocks[0].mlp.wdown.lowbit_runtime_state.hadamard)


if __name__ == "__main__":
    unittest.main()
