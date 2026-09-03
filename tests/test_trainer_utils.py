from __future__ import annotations

import math

from torch.utils.data import SequentialSampler

from trainer.trainer_utils import SkipBatchSampler, get_lr


def test_cosine_schedule_boundaries():
    base_lr = 1e-3
    assert math.isclose(get_lr(0, 100, base_lr), base_lr)
    assert math.isclose(get_lr(100, 100, base_lr), base_lr * 0.1)


def test_skip_batch_sampler_resumes_on_batch_boundary():
    sampler = SequentialSampler(range(10))
    batches = list(SkipBatchSampler(sampler, batch_size=3, skip_batches=2))

    assert batches == [[6, 7, 8], [9]]
    assert len(batches) == 2

