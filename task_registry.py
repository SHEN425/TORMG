TASK_NAME_TO_ID = {
    "coverage_hole_detection": 0,
}

TASK_ID_TO_NAME = {task_id: name for name, task_id in TASK_NAME_TO_ID.items()}


def get_num_tasks() -> int:
    return len(TASK_NAME_TO_ID)


def get_task_name(task_id: int) -> str:
    if task_id not in TASK_ID_TO_NAME:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASK_ID_TO_NAME[task_id]


def get_task_id(task_name: str) -> int:
    if task_name not in TASK_NAME_TO_ID:
        raise KeyError(f"Unknown task_name: {task_name}")
    return TASK_NAME_TO_ID[task_name]


def resolve_task(config: dict):
    if "task_name" in config:
        task_name = config["task_name"]
        task_id = get_task_id(task_name)
        return task_name, task_id

    if "task_id" in config:
        task_id = int(config["task_id"])
        task_name = get_task_name(task_id)
        return task_name, task_id

    # Backward-compatible default.
    task_name = "coverage_hole_detection"
    task_id = get_task_id(task_name)
    return task_name, task_id
