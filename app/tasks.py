TASK_CONFIGS = {
    "easy": {
        "max_steps": 3,
        "stages": ["screening"],
        "num_candidates": 1,
    },
    "medium": {
        "max_steps": 5,
        "stages": ["screening", "interview"],
        "num_candidates": 1,
    },
    "hard": {
        "max_steps": 8,
        "stages": ["screening", "interview", "hr_decision"],
        "num_candidates": 3,
    },
}

DEFAULT_TASK_ORDER = ["easy", "medium", "hard"]
