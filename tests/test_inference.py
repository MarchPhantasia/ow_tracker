from __future__ import annotations

from src.head_tracker.inference import class_id_by_name, selected_class_ids


def test_selected_class_ids_defaults_to_enemy_only():
    names = {0: "ally", 1: "enemy"}

    assert selected_class_ids(names) == [1]


def test_selected_class_ids_can_include_ally():
    names = {0: "ally", 1: "enemy"}

    assert selected_class_ids(names, include_ally=True) is None


def test_class_id_by_name_is_case_insensitive():
    assert class_id_by_name({0: "Ally", 1: "Enemy"}, "enemy") == 1
