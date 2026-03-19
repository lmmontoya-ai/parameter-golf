from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    torch = None
    IMPORT_ERROR = f"torch is required for tokenizer/eval mainline tests: {exc}"
else:
    IMPORT_ERROR = ""

if torch is not None:
    import train_gpt as tg


TOKENIZER_SPECS_PATH = Path(__file__).resolve().parents[1] / "data" / "tokenizer_specs.json"


class TokenizerSpecsTest(unittest.TestCase):
    def test_specs_include_sp2048_and_sp4096(self):
        payload = json.loads(TOKENIZER_SPECS_PATH.read_text(encoding="utf-8"))
        tokenizers = {spec["name"]: spec for spec in payload["tokenizers"]}

        self.assertIn("sp_bpe_2048", tokenizers)
        self.assertIn("sp_bpe_4096", tokenizers)
        self.assertEqual(tokenizers["sp_bpe_2048"]["dataset_suffix"], "sp2048")
        self.assertEqual(tokenizers["sp_bpe_2048"]["vocab_size"], 2048)
        self.assertEqual(tokenizers["sp_bpe_4096"]["dataset_suffix"], "sp4096")
        self.assertEqual(tokenizers["sp_bpe_4096"]["vocab_size"], 4096)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class EvalWindowPlanTest(unittest.TestCase):
    def test_stride_equal_seq_len_matches_contiguous_mode(self):
        contiguous = tg.build_eval_plan(total_targets=32, eval_seq_len=8, stride=0)
        stride_equal = tg.build_eval_plan(total_targets=32, eval_seq_len=8, stride=8)

        self.assertEqual(contiguous.mode, "contiguous")
        self.assertEqual(stride_equal.mode, "contiguous")
        self.assertEqual(contiguous.total_windows, stride_equal.total_windows)
        self.assertEqual(contiguous.scored_tokens, stride_equal.scored_tokens)

        contiguous_windows = [tg.eval_window_for_index(contiguous, i) for i in range(contiguous.total_windows)]
        stride_windows = [tg.eval_window_for_index(stride_equal, i) for i in range(stride_equal.total_windows)]
        self.assertEqual(contiguous_windows, stride_windows)

    def test_overlapping_windows_score_each_target_once(self):
        plan = tg.build_eval_plan(total_targets=25, eval_seq_len=8, stride=3)
        covered = torch.zeros((25,), dtype=torch.int32)

        for idx in range(plan.total_windows):
            start, score_offset, score_len = tg.eval_window_for_index(plan, idx)
            covered[start + score_offset : start + score_offset + score_len] += 1

        self.assertEqual(plan.mode, "sliding")
        self.assertEqual(int(covered.sum().item()), 25)
        self.assertTrue(torch.all(covered == 1).item())


@unittest.skipIf(torch is None, IMPORT_ERROR)
class EvalValSlidingTest(unittest.TestCase):
    class DummyModel(nn.Module):
        def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
            del input_ids
            losses = target_ids.float()
            if loss_mask is not None:
                losses = losses[loss_mask]
            return losses.mean()

    def make_args(self, *, eval_seq_len: int, stride: int):
        return SimpleNamespace(
            val_batch_size=64,
            train_seq_len=8,
            eval_seq_len=eval_seq_len,
            eval_window_stride=stride,
        )

    def test_eval_val_stride_equal_seq_len_matches_contiguous(self):
        model = self.DummyModel()
        device = torch.device("cpu")
        val_tokens = torch.arange(17, dtype=torch.int64)
        base_bytes_lut = torch.ones((32,), dtype=torch.int16)
        has_leading_space_lut = torch.zeros((32,), dtype=torch.bool)
        is_boundary_token_lut = torch.zeros((32,), dtype=torch.bool)

        loss_a, bpb_a = tg.eval_val(
            self.make_args(eval_seq_len=8, stride=0),
            model,
            0,
            1,
            device,
            1,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        loss_b, bpb_b = tg.eval_val(
            self.make_args(eval_seq_len=8, stride=8),
            model,
            0,
            1,
            device,
            1,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )

        self.assertAlmostEqual(loss_a, loss_b, places=6)
        self.assertAlmostEqual(bpb_a, bpb_b, places=6)

    def test_eval_val_sliding_scores_all_targets_once(self):
        model = self.DummyModel()
        device = torch.device("cpu")
        val_tokens = torch.arange(26, dtype=torch.int64)
        base_bytes_lut = torch.ones((32,), dtype=torch.int16)
        has_leading_space_lut = torch.zeros((32,), dtype=torch.bool)
        is_boundary_token_lut = torch.zeros((32,), dtype=torch.bool)

        loss, bpb = tg.eval_val(
            self.make_args(eval_seq_len=8, stride=3),
            model,
            0,
            1,
            device,
            1,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )

        expected_loss = float(torch.arange(1, 26, dtype=torch.float32).mean().item())
        expected_bpb = expected_loss / math.log(2.0)

        self.assertAlmostEqual(loss, expected_loss, places=6)
        self.assertAlmostEqual(bpb, expected_bpb, places=6)


@unittest.skipIf(torch is None, IMPORT_ERROR)
class FactorizedTiedEmbeddingTest(unittest.TestCase):
    def make_model(self, *, factor_embed_enable: bool = False, tie_embeddings: bool = True, factor_embed_dim: int = 16):
        return tg.GPT(
            vocab_size=32,
            num_layers=4,
            model_dim=32,
            num_heads=4,
            num_kv_heads=2,
            mlp_mult=2,
            tie_embeddings=tie_embeddings,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            attnres_enable=False,
            attnres_block_layers=3,
            factor_embed_enable=factor_embed_enable,
            factor_embed_dim=factor_embed_dim,
        )

    def test_factorized_embedding_matches_tied_contract(self):
        model = self.make_model(factor_embed_enable=True, factor_embed_dim=16)
        input_ids = torch.randint(0, 32, (2, 8))
        target_ids = torch.randint(0, 32, (2, 8))

        loss = model(input_ids, target_ids)
        loss.backward()

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(model(input_ids, target_ids).shape, torch.Size([]))
        self.assertEqual(model.factor_embed_param_count(), 32 * 16 + 32 * 16)

    def test_factorized_embedding_requires_tied_embeddings(self):
        with self.assertRaisesRegex(ValueError, "requires TIE_EMBEDDINGS=1"):
            _ = self.make_model(factor_embed_enable=True, tie_embeddings=False)

    def test_factorized_embedding_roundtrip_state_dict(self):
        model = self.make_model(factor_embed_enable=True, factor_embed_dim=16)
        quant_obj, _ = tg.quantize_state_dict_int8(model.state_dict())
        restored = tg.dequantize_state_dict_int8(quant_obj)
        missing, unexpected = model.load_state_dict(restored, strict=False)

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()
