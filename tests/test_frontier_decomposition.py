from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = f"torch is required for frontier decomposition tests: {exc}"
else:
    IMPORT_ERROR = ""

if torch is not None:
    import train_gpt as tg


@unittest.skipIf(torch is None, IMPORT_ERROR)
class NorMuonTest(unittest.TestCase):
    def test_normuon_tracks_row_second_moment(self):
        param = torch.nn.Parameter(torch.randn(4, 3))
        opt = tg.Muon(
            [param],
            lr=0.01,
            momentum=0.95,
            backend_steps=1,
            variant="normuon",
            norm_beta2=0.95,
        )

        param.grad = torch.randn_like(param)
        opt.step()

        state = opt.state[param]
        self.assertIn("row_second_moment", state)
        self.assertEqual(state["row_second_moment"].shape, torch.Size([4]))
        self.assertEqual(state["row_second_moment"].dtype, torch.float32)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class FakeQuantLinearTest(unittest.TestCase):
    def test_casted_linear_fake_quant_forward_backward(self):
        linear = tg.CastedLinear(8, 4, bias=False)
        tg.configure_model_fake_quant(linear, enabled=True, quant_bits=6, clip_quantile=None)

        x = torch.randn(2, 8)
        loss = linear(x).sum()
        loss.backward()

        self.assertIsNotNone(linear.weight.grad)
        self.assertTrue(linear.fake_quant_enabled)

    def test_temporarily_disable_fake_quant_restores_state(self):
        linear = tg.CastedLinear(8, 4, bias=False)
        tg.configure_model_fake_quant(linear, enabled=True, quant_bits=6, clip_quantile=None)

        with tg.temporarily_disable_fake_quant(linear):
            self.assertFalse(linear.fake_quant_enabled)
        self.assertTrue(linear.fake_quant_enabled)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class Int6ExportTest(unittest.TestCase):
    def test_int6_quantization_bounds_and_restore(self):
        state_dict = {
            "weight": torch.randn(1024, 128),
            "bias": torch.randn(16),
        }
        quant_obj, _ = tg.quantize_state_dict(
            state_dict,
            export_format="int6_zstd",
            passthrough_fp16_patterns=(),
        )
        self.assertLessEqual(int(quant_obj["quantized"]["weight"].abs().max().item()), 31)

        restored = tg.dequantize_state_dict(quant_obj)
        self.assertEqual(restored["weight"].dtype, torch.float32)
        self.assertEqual(restored["bias"].dtype, torch.float32)
        self.assertEqual(restored["weight"].shape, state_dict["weight"].shape)

    def test_int6_zstd_roundtrip_or_clear_dependency_failure(self):
        state_dict = {"weight": torch.randn(1024, 128)}
        quant_obj, _ = tg.quantize_state_dict(
            state_dict,
            export_format="int6_zstd",
            passthrough_fp16_patterns=(),
        )
        if tg.zstd is None:
            with self.assertRaisesRegex(RuntimeError, "zstandard"):
                tg.serialize_quantized_obj(quant_obj, export_format="int6_zstd")
            return

        blob, _ = tg.serialize_quantized_obj(quant_obj, export_format="int6_zstd")
        restored_obj = tg.deserialize_quantized_obj(blob, export_format="int6_zstd")
        restored = tg.dequantize_state_dict(restored_obj)
        self.assertEqual(restored["weight"].shape, state_dict["weight"].shape)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class SwaAccumulatorTest(unittest.TestCase):
    def test_swa_accumulator_truncates_and_averages(self):
        acc = tg.SwaAccumulator(max_checkpoints=2)
        acc.add(10, {"weight": torch.tensor([1.0, 3.0]), "token": torch.tensor([1], dtype=torch.int64)})
        acc.add(20, {"weight": torch.tensor([3.0, 5.0]), "token": torch.tensor([2], dtype=torch.int64)})
        acc.add(30, {"weight": torch.tensor([5.0, 7.0]), "token": torch.tensor([3], dtype=torch.int64)})

        self.assertEqual(len(acc), 2)
        self.assertEqual(acc.retained_steps(), [20, 30])
        avg = acc.average_state_dict()
        self.assertTrue(torch.allclose(avg["weight"], torch.tensor([4.0, 6.0])))
        self.assertEqual(int(avg["token"].item()), 3)

    def test_surrogate_swa_retains_best_scores(self):
        acc = tg.SwaAccumulator(max_checkpoints=2, select_mode="surrogate_roundtrip")
        acc.add(100, {"weight": torch.tensor([1.0, 3.0])}, surrogate_val_bpb=1.30)
        acc.add(200, {"weight": torch.tensor([3.0, 5.0])}, surrogate_val_bpb=1.10)
        acc.add(300, {"weight": torch.tensor([5.0, 7.0])}, surrogate_val_bpb=1.20)

        self.assertEqual(len(acc), 2)
        self.assertEqual(acc.retained_step_scores(), [(200, 1.1), (300, 1.2)])
        avg = acc.average_state_dict()
        self.assertTrue(torch.allclose(avg["weight"], torch.tensor([4.0, 6.0])))

    def test_resolve_swa_start_step(self):
        args = SimpleNamespace(iterations=20_000, warmdown_iters=3_000, swa_start_frac=0.5)
        self.assertEqual(tg.resolve_swa_start_step(args), 18_500)

    def test_should_collect_swa_uses_wallclock_when_present(self):
        args = SimpleNamespace(swa_enabled=True, swa_every=200, warmdown_iters=3_000, swa_start_frac=0.5)
        self.assertFalse(
            tg.should_collect_swa(
                8_000,
                args,
                18_500,
                elapsed_ms=800_000.0,
                max_wallclock_ms=1_200_000.0,
            )
        )
        self.assertTrue(
            tg.should_collect_swa(
                9_600,
                args,
                18_500,
                elapsed_ms=1_080_000.0,
                max_wallclock_ms=1_200_000.0,
            )
        )


@unittest.skipIf(torch is None, IMPORT_ERROR)
class SurrogateValidationTokensTest(unittest.TestCase):
    def test_surrogate_tokens_truncate_to_requested_target_count(self):
        tokens = torch.arange(0, 10_000, dtype=torch.int64)
        truncated = tg.truncate_validation_tokens_for_surrogate(tokens, max_targets=4_096, eval_seq_len=1_024)
        self.assertEqual(truncated.numel(), 4_097)

    def test_surrogate_tokens_keep_minimum_eval_length(self):
        tokens = torch.arange(0, 10_000, dtype=torch.int64)
        truncated = tg.truncate_validation_tokens_for_surrogate(tokens, max_targets=128, eval_seq_len=1_024)
        self.assertEqual(truncated.numel(), 1_025)


if __name__ == "__main__":
    unittest.main()
