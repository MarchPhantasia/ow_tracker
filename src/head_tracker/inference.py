from __future__ import annotations


def class_id_by_name(names: dict[int, str], class_name: str) -> int:
    target = class_name.strip().lower()
    for class_id, name in names.items():
        if name.strip().lower() == target:
            return class_id
    raise KeyError(f"class {class_name!r} not found in {names}")


def selected_class_ids(
    names: dict[int, str],
    *,
    include_ally: bool = False,
    target_class: str = "enemy",
) -> list[int] | None:
    if include_ally:
        return None
    return [class_id_by_name(names, target_class)]
