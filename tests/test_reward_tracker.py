from recipe.hint.reward_tracker import RewardTracker


def test_log_hint_payloads_normalizes_tuple_batch_keys():
    tracker = RewardTracker()

    tracker.log_hint_payloads(
        ["sample-0", "sample-1"],
        {(1, "question", "solution"): {"level_1": "use symmetry"}},
        global_step=7,
        used=False,
        failed=True,
    )

    assert tracker.index_to_hint_history["sample-1"] == [
        {
            "step": 7,
            "level": "level_1",
            "hint": "use symmetry",
            "used": False,
            "failed": True,
        }
    ]


def test_log_hint_raw_normalizes_tuple_batch_keys():
    tracker = RewardTracker()

    tracker.log_hint_raw(
        ["sample-0", "sample-1"],
        {(0, "question", "solution"): "raw hint payload"},
        global_step=11,
        used=False,
        failed=True,
    )

    assert tracker.index_to_hint_raw_history["sample-0"] == [
        {
            "step": 11,
            "raw": "raw hint payload",
            "used": False,
            "failed": True,
        }
    ]


def test_log_hint_payloads_ignores_invalid_tuple_keys():
    tracker = RewardTracker()

    tracker.log_hint_payloads(
        ["sample-0"],
        {("not-an-index", "question"): {"level_1": "ignored"}},
        global_step=1,
        used=False,
    )

    assert tracker.index_to_hint_history == {}
