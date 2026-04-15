"""Unit tests for compressed logits sampling in distributed speculative decoding."""

import torch
import torch.nn.functional as F
from tests.ut.base import TestBase

from vllm.v1.sample.metadata import SamplingMetadata
from vllm_ascend.sample.rejection_sampler import (
    apply_sampling_constraints,
    rejection_sample,
)


class TestCompressedSampling(TestBase):
    """Test compressed logits sampling flow."""

    def setUp(self):
        """Set up test parameters."""
        super().setUp()
        self.batch_size = 2
        self.num_draft_tokens = [3, 2]  # Different number of draft tokens per request
        self.max_spec_len = 4
        self.vocab_size = 1000  # Full vocabulary size
        self.top_k = 20  # Top-k for sampling
        self.tp_size = 2  # Tensor parallel size
        self.compressed_vocab_size = self.top_k * self.tp_size  # 40

    def test_apply_sampling_constraints_compressed_mode(self):
        """Test apply_sampling_constraints with compressed logits."""
        # Simulate local logits from each TP rank (before allgather)
        # Shape: [num_tokens, vocab_size // tp_size]
        num_tokens = sum(self.num_draft_tokens)  # 5 tokens total
        local_vocab_size = self.vocab_size // self.tp_size

        # Create mock logits (before processing)
        logits = torch.randn(num_tokens, local_vocab_size, dtype=torch.float32)

        # Create cumulative draft tokens
        cu_num_draft_tokens = torch.tensor(
            [self.num_draft_tokens[0], sum(self.num_draft_tokens)],
            dtype=torch.int32
        )

        # Create sampling metadata
        temperature = torch.tensor([1.0, 1.0], dtype=torch.float32)
        generators = {0: torch.Generator().manual_seed(42)}

        sampling_metadata = SamplingMetadata(
            temperature=temperature,
            all_greedy=False,
            all_random=True,
            generators=generators,
            no_penalties=True,
        )

        # Apply sampling constraints
        processed_logits, target_indices = apply_sampling_constraints(
            logits,
            cu_num_draft_tokens,
            sampling_metadata,
        )

        # Verify output shapes
        self.assertIsNotNone(target_indices)
        self.assertEqual(processed_logits.shape[0], num_tokens)
        self.assertEqual(processed_logits.shape[1], self.compressed_vocab_size)
        self.assertEqual(target_indices.shape[0], num_tokens)
        self.assertEqual(target_indices.shape[1], self.compressed_vocab_size)

        # Verify indices are within vocabulary range
        self.assertTrue((target_indices >= 0).all())
        self.assertTrue((target_indices < self.vocab_size).all())

        # Verify logits are processed (should be probabilities after softmax)
        # Note: processed_logits should be logits, not probabilities yet
        self.assertTrue(torch.isfinite(processed_logits).all())

    def test_rejection_sample_compressed_mode(self):
        """Test rejection sampling with compressed logits."""
        num_tokens = sum(self.num_draft_tokens)

        # Create mock draft token IDs (global vocabulary indices)
        draft_token_ids = torch.randint(
            0, self.vocab_size, (num_tokens,), dtype=torch.int64
        )

        # Create cumulative draft tokens
        cu_num_draft_tokens = torch.tensor(
            [self.num_draft_tokens[0], sum(self.num_draft_tokens)],
            dtype=torch.int32
        )

        # Create mock draft probabilities (full vocabulary)
        draft_probs = F.softmax(
            torch.randn(num_tokens, self.vocab_size, dtype=torch.float32), dim=-1
        )

        # Create mock target logits (compressed form)
        # Shape: [num_tokens, compressed_vocab_size]
        target_logits = torch.randn(
            num_tokens, self.compressed_vocab_size, dtype=torch.float32
        )

        # Create mock target indices (global vocabulary indices)
        # Shape: [num_tokens, compressed_vocab_size]
        target_indices = torch.randint(
            0, self.vocab_size, (num_tokens, self.compressed_vocab_size), dtype=torch.int64
        )

        # Ensure some draft tokens are in the target candidates for acceptance
        for i in range(num_tokens):
            # Put draft token at position 0 in candidates
            target_indices[i, 0] = draft_token_ids[i]

        # Create bonus token IDs
        bonus_token_ids = torch.randint(
            0, self.vocab_size, (self.batch_size,), dtype=torch.int64
        )

        # Create sampling metadata
        temperature = torch.tensor([1.0, 1.0], dtype=torch.float32)
        generators = {
            0: torch.Generator().manual_seed(42),
            1: torch.Generator().manual_seed(123),
        }

        sampling_metadata = SamplingMetadata(
            temperature=temperature,
            all_greedy=False,
            all_random=True,
            generators=generators,
            no_penalties=True,
        )

        # Run rejection sampling
        output_token_ids = rejection_sample(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=self.num_draft_tokens,
            max_spec_len=self.max_spec_len,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_probs=draft_probs,
            target_logits_or_tuple=(target_logits, target_indices),
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
        )

        # Verify output shape
        self.assertEqual(output_token_ids.shape[0], self.batch_size)
        self.assertEqual(output_token_ids.shape[1], self.max_spec_len + 1)

        # Verify output tokens are valid vocabulary indices
        self.assertTrue((output_token_ids >= 0).all() or (output_token_ids == -1).all())
        valid_mask = output_token_ids >= 0
        self.assertTrue((output_token_ids[valid_mask] < self.vocab_size).all())

    def test_rejection_sample_with_ngram(self):
        """Test rejection sampling with N-GRAM (no draft probs)."""
        num_tokens = sum(self.num_draft_tokens)

        # Create mock draft token IDs
        draft_token_ids = torch.randint(
            0, self.vocab_size, (num_tokens,), dtype=torch.int64
        )

        # Create cumulative draft tokens
        cu_num_draft_tokens = torch.tensor(
            [self.num_draft_tokens[0], sum(self.num_draft_tokens)],
            dtype=torch.int32
        )

        # N-GRAM case: draft_probs is None
        draft_probs = None

        # Create mock target logits (compressed form)
        target_logits = torch.randn(
            num_tokens, self.compressed_vocab_size, dtype=torch.float32
        )

        # Create mock target indices
        target_indices = torch.randint(
            0, self.vocab_size, (num_tokens, self.compressed_vocab_size), dtype=torch.int64
        )

        # Create bonus token IDs
        bonus_token_ids = torch.randint(
            0, self.vocab_size, (self.batch_size,), dtype=torch.int64
        )

        # Create sampling metadata
        temperature = torch.tensor([1.0, 1.0], dtype=torch.float32)
        generators = {
            0: torch.Generator().manual_seed(42),
            1: torch.Generator().manual_seed(123),
        }

        sampling_metadata = SamplingMetadata(
            temperature=temperature,
            all_greedy=False,
            all_random=True,
            generators=generators,
            no_penalties=True,
        )

        # Run rejection sampling
        output_token_ids = rejection_sample(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=self.num_draft_tokens,
            max_spec_len=self.max_spec_len,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_probs=draft_probs,
            target_logits_or_tuple=(target_logits, target_indices),
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
        )

        # Verify output shape
        self.assertEqual(output_token_ids.shape[0], self.batch_size)
        self.assertEqual(output_token_ids.shape[1], self.max_spec_len + 1)

    def test_full_sampling_pipeline(self):
        """Test complete sampling pipeline from raw logits to output tokens."""
        num_tokens = sum(self.num_draft_tokens)
        local_vocab_size = self.vocab_size // self.tp_size

        # Step 1: Simulate raw logits from model (before allgather)
        raw_logits = torch.randn(num_tokens, local_vocab_size, dtype=torch.float32)

        # Step 2: Apply sampling constraints (top-k -> allgather -> top-p)
        cu_num_draft_tokens = torch.tensor(
            [self.num_draft_tokens[0], sum(self.num_draft_tokens)],
            dtype=torch.int32
        )

        temperature = torch.tensor([1.0, 1.0], dtype=torch.float32)
        generators = {
            0: torch.Generator().manual_seed(42),
            1: torch.Generator().manual_seed(123),
        }

        sampling_metadata = SamplingMetadata(
            temperature=temperature,
            all_greedy=False,
            all_random=True,
            generators=generators,
            no_penalties=True,
        )

        processed_logits, target_indices = apply_sampling_constraints(
            raw_logits,
            cu_num_draft_tokens,
            sampling_metadata,
        )

        # Verify processed output
        self.assertIsNotNone(target_indices)
        self.assertEqual(processed_logits.shape[1], self.compressed_vocab_size)

        # Step 3: Run rejection sampling
        draft_token_ids = torch.randint(
            0, self.vocab_size, (num_tokens,), dtype=torch.int64
        )

        draft_probs = F.softmax(
            torch.randn(num_tokens, self.vocab_size, dtype=torch.float32), dim=-1
        )

        bonus_token_ids = torch.randint(
            0, self.vocab_size, (self.batch_size,), dtype=torch.int64
        )

        output_token_ids = rejection_sample(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=self.num_draft_tokens,
            max_spec_len=self.max_spec_len,
            cu_num_draft_tokens=cu_num_draft_tokens,
            draft_probs=draft_probs,
            target_logits_or_tuple=(processed_logits, target_indices),
            bonus_token_ids=bonus_token_ids,
            sampling_metadata=sampling_metadata,
        )

        # Verify final output
        self.assertEqual(output_token_ids.shape, (self.batch_size, self.max_spec_len + 1))
        print(f"\nFinal output tokens shape: {output_token_ids.shape}")
        print(f"Output tokens:\n{output_token_ids}")

    def test_greedy_sampling_compressed_mode(self):
        """Test greedy sampling with compressed logits."""
        num_tokens = sum(self.num_draft_tokens)
        local_vocab_size = self.vocab_size // self.tp_size

        # Create logits where greedy sampling is used
        raw_logits = torch.randn(num_tokens, local_vocab_size, dtype=torch.float32)

        cu_num_draft_tokens = torch.tensor(
            [self.num_draft_tokens[0], sum(self.num_draft_tokens)],
            dtype=torch.int32
        )

        # All greedy sampling
        temperature = torch.tensor([0.0, 0.0], dtype=torch.float32)  # 0 means greedy

        sampling_metadata = SamplingMetadata(
            temperature=temperature,
            all_greedy=True,
            all_random=False,
            generators={},
            no_penalties=True,
        )

        # Apply sampling constraints
        processed_logits, target_indices = apply_sampling_constraints(
            raw_logits,
            cu_num_draft_tokens,
            sampling_metadata,
        )

        # For greedy sampling, target_indices should be None
        self.assertIsNone(target_indices)
        # processed_logits should have local vocab size
        self.assertEqual(processed_logits.shape[1], local_vocab_size)


if __name__ == "__main__":
    import unittest
    unittest.main()