# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
from pathlib import Path


INITIALIZE_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "megatron"
    / "bridge"
    / "training"
    / "initialize.py"
)


def _call_name(call: ast.Call) -> str:
    parts = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _calls(node: ast.AST) -> list[str]:
    return [_call_name(item) for item in ast.walk(node) if isinstance(item, ast.Call)]


def test_dataset_helpers_are_loaded_on_every_initialized_rank_without_a_barrier():
    tree = ast.parse(INITIALIZE_SOURCE.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "megatron.core.datasets.utils"
        for alias in node.names
    }
    assert "compile_helpers_distributed" in imported_names

    helper = functions["_compile_dataset_helpers"]
    helper_calls = _calls(helper)
    assert helper_calls.count("compile_helpers_distributed") == 1
    assert "torch.distributed.get_rank" in helper_calls
    assert "torch.distributed.barrier" not in helper_calls

    initialize = functions["initialize_megatron"]
    guarded_calls = []
    for conditional in (node for node in ast.walk(initialize) if isinstance(node, ast.If)):
        if "torch.distributed.is_initialized" in _calls(conditional.test):
            guarded_calls.extend(
                call_name
                for statement in conditional.body
                for call_name in _calls(statement)
            )
    assert guarded_calls.count("_compile_dataset_helpers") == 1
