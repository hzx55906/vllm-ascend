# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host dispatch checks without importing PyTorch, Triton or the NPU plugin.

Device correctness is covered by test_v2_paga_* in the Triton nightly tests.
This file can also be run directly with unittest (requires pydantic).
"""

import __future__
import ast
import importlib.util
import math

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from pydantic import StrictBool, model_validator
from pydantic.dataclasses import rebuild_dataclass


class TestPagaDispatch(unittest.TestCase):
    def setUp(self):
        self.path = (
            Path(__file__).resolve().parents[3]
            / "vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py"
        )
        self.tree = ast.parse(self.path.read_text(encoding="utf-8"))

    def run_dispatch(self, enabled, **overrides):
        # Execute the actual Python entry point with mocked allocations and
        # launches; do not attempt to emulate device-side acceptance here.
        function = next(
            n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "rejection_sample"
        )
        torch = SimpleNamespace(argmax=Mock(), int64="int64", int32="int32", float32="float32")
        kernels = {
            name: MagicMock()
            for name in (
                "_compute_block_stats_kernel",
                "_probabilistic_rejection_kernel",
                "resample",
            )
        }
        namespace = {
            "torch": torch,
            "get_ascend_config": lambda: SimpleNamespace(
                rejection_sampler_config=SimpleNamespace(
                    probability_augmented_greedy_acceptance=enabled,
                ),
            ),
            "triton": SimpleNamespace(
                cdiv=lambda a, b: (a + b - 1) // b,
                next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            ),
            **kernels,
        }
        code = compile(
            ast.Module(body=[function], type_ignores=[]),
            str(self.path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        )
        exec(code, namespace)
        target = MagicMock(shape=(6, 9000))
        cu_num_logits = MagicMock(shape=(3,))
        inputs = dict(
            target_logits=target,
            draft_logits=None,
            draft_sampled=MagicMock(),
            cu_num_logits=cu_num_logits,
            pos=MagicMock(),
            idx_mapping=MagicMock(),
            expanded_idx_mapping=MagicMock(),
            expanded_local_pos=MagicMock(),
            temperature=MagicMock(),
            seed=MagicMock(),
            num_speculative_steps=2,
        )
        inputs.update(overrides)
        namespace["rejection_sample"](**inputs)
        return torch, kernels, target

    def test_disabled_does_not_compute_greedy_tokens(self):
        torch, kernels, _ = self.run_dispatch(False)
        torch.argmax.assert_not_called()
        launch = kernels["_probabilistic_rejection_kernel"].__getitem__.return_value
        self.assertIsNone(launch.call_args.args[7])
        self.assertFalse(launch.call_args.kwargs["PAGA_ENABLED"])

    def test_enabled_passes_argmax_to_acceptance_only(self):
        torch, kernels, target = self.run_dispatch(True)
        torch.argmax.assert_called_once_with(target.__getitem__.return_value, dim=-1)
        launch = kernels["_probabilistic_rejection_kernel"].__getitem__.return_value
        self.assertIs(launch.call_args.args[7], torch.argmax.return_value)
        self.assertTrue(launch.call_args.kwargs["PAGA_ENABLED"])
        stats_launch = kernels["_compute_block_stats_kernel"].__getitem__.return_value
        self.assertNotIn("PAGA_ENABLED", stats_launch.call_args.kwargs)


    def test_block_verification_cannot_silently_enable_paga(self):
        with self.assertRaisesRegex(ValueError, "not block"):
            self.run_dispatch(True, use_block_verification=True)

    def test_synthetic_conflict_and_fp64(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            self.run_dispatch(True, synthetic_conditional_rates=object())
        with self.assertRaises(NotImplementedError):
            self.run_dispatch(True, use_fp64=True)

    def test_disabled_preserves_synthetic_mode(self):
        torch, kernels, _ = self.run_dispatch(False, synthetic_conditional_rates=object())
        torch.argmax.assert_not_called()
        launch = kernels["_probabilistic_rejection_kernel"].__getitem__.return_value
        self.assertTrue(launch.call_args.kwargs["SYNTHETIC_MODE"])

    def test_argmax_uses_effective_draft_vocabulary(self):
        draft = MagicMock()
        draft.size.return_value = 8000
        torch, _, target = self.run_dispatch(True, draft_logits=draft)
        target.__getitem__.assert_called_once_with((slice(None), slice(None, 8000)))
        torch.argmax.assert_called_once_with(target.__getitem__.return_value, dim=-1)

    def test_legacy_environment_does_not_enable_paga(self):
        # The PR has one control source: the engine's additional_config.
        with patch.dict("os.environ", {"VLLM_ASCEND_ENABLE_LOSSY_SPEC_ACCEPTANCE": "1"}):
            torch, _, _ = self.run_dispatch(False)
        torch.argmax.assert_not_called()

    def test_config_true_is_not_masked_by_legacy_environment(self):
        with patch.dict("os.environ", {"VLLM_ASCEND_ENABLE_LOSSY_SPEC_ACCEPTANCE": "0"}):
            torch, _, _ = self.run_dispatch(True)
        torch.argmax.assert_called_once()

    def test_statistics_kernel_is_imported_not_reimplemented(self):
        self.assertFalse(
            any(isinstance(n, ast.FunctionDef) and n.name == "_compute_block_stats_kernel" for n in self.tree.body)
        )
        imports = [
            alias
            for n in self.tree.body
            if isinstance(n, ast.ImportFrom)
            and n.module == "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils"
            for alias in n.names
        ]
        self.assertTrue(
            any(
                alias.name == "_compute_local_logits_stats_kernel" and alias.asname == "_compute_block_stats_kernel"
                for alias in imports
            )
        )


class TestPagaConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ascend_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RejectionSamplerConfig")
        spec = importlib.util.spec_from_file_location("config_utils", path.parent / "config_utils.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = {
            "config": module.config,
            "StrictBool": StrictBool,
            "model_validator": model_validator,
        }
        exec(
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                str(path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        cls.config_type = namespace["RejectionSamplerConfig"]
        rebuild_dataclass(cls.config_type, _types_namespace=namespace)

    def test_default_and_explicit_false_are_disabled(self):
        for config in (None, {}, {"probability_augmented_greedy_acceptance": False}):
            with self.subTest(config=config):
                parsed = self.config_type(**(config or {}))
                self.assertFalse(parsed.probability_augmented_greedy_acceptance)
                # A disabled policy imposes no new requirements on engines.
                parsed.validate_for_vllm_config(SimpleNamespace())


    def test_only_actual_booleans_are_accepted(self):
        for value in (0, 1, "true", "false", None, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "valid boolean"):
                self.config_type(probability_augmented_greedy_acceptance=value)

    def test_conflicting_ascend_verifiers_are_rejected(self):
        for name in ("enable_block_verify", "enable_entropy_verify"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "cannot be combined"):
                self.config_type(**{"probability_augmented_greedy_acceptance": True, name: True})

    def test_v2_standard_speculation_is_supported(self):
        config = self.config_type(probability_augmented_greedy_acceptance=True)
        config.validate_for_vllm_config(
            SimpleNamespace(
                speculative_config=SimpleNamespace(rejection_sample_method="standard"),
                use_v2_model_runner=True,
            )
        )

    def test_config_instances_do_not_leak_the_switch(self):
        enabled = self.config_type(probability_augmented_greedy_acceptance=True)
        disabled = self.config_type()
        self.assertTrue(enabled.probability_augmented_greedy_acceptance)
        self.assertFalse(disabled.probability_augmented_greedy_acceptance)

    def test_unsupported_engine_modes_are_rejected(self):
        config = self.config_type(probability_augmented_greedy_acceptance=True)
        cases = [
            (None, True, "requires speculative decoding"),
            (SimpleNamespace(rejection_sample_method="standard"), False, "v2 model runner"),
            (SimpleNamespace(rejection_sample_method="block"), True, "requires rejection_sample_method"),
            (SimpleNamespace(rejection_sample_method="synthetic"), True, "requires rejection_sample_method"),
        ]
        for spec, v2, error in cases:
            with self.subTest(spec=spec, v2=v2), self.assertRaisesRegex(ValueError, error):
                config.validate_for_vllm_config(SimpleNamespace(speculative_config=spec, use_v2_model_runner=v2))


class TestPagaAcceptanceMath(unittest.TestCase):
    """Execute the actual scalar acceptance branch with scalar tl operations.

    This checks probability boundaries without NPU dependencies; device tests
    still cover Triton compilation, RNG, and resampling.
    """

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        branch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "PAGA_ENABLED"
        )
        cls.code = compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec")

    def accepts(self, p, q, u, enabled, match=True):
        state = {
            "tl": SimpleNamespace(log=math.log, load=lambda _: 7),
            "PAGA_ENABLED": enabled,
            "accepted": True,
            "target_log_prob": math.log(p) if p else -math.inf,
            "draft_log_prob": math.log(q),
            "u": u,
            "draft_sampled": 7 if match else 8,
            "greedy_token_ids_ptr": 0,
            "logit_idx": 0,
        }
        exec(self.code, state)
        return bool(state["accepted"])

    def test_probability_mass_for_one_hot_and_probabilistic_drafts(self):
        # Deterministic uniform grid checks actual probability mass, not one seed.
        for p, q in ((0.6, 1.0), (0.6, 0.9), (0.8, 0.2), (0.0, 1.0)):
            for enabled in (False, True):
                for match in (False, True):
                    with self.subTest(p=p, q=q, enabled=enabled, match=match):
                        a = min(1.0, p / q)
                        expected = 1.0 if enabled and match else a
                        observed = (
                            sum(self.accepts(p, q, (i + 0.5) / 1000, enabled, match) for i in range(1000)) / 1000
                        )
                        self.assertAlmostEqual(observed, expected, delta=0.001)

    def test_enabled_forces_argmax_acceptance(self):
        self.assertFalse(self.accepts(0.6, 1.0, 0.99, False))
        self.assertTrue(self.accepts(0.6, 1.0, 0.99, True))
        self.assertFalse(self.accepts(0.6, 1.0, 0.99, True, match=False))

    def test_large_probability_ratio_does_not_overflow(self):
        self.assertTrue(self.accepts(0.6, 1e-300, 0.99, True))


if __name__ == "__main__":
    unittest.main()
