"""Shared fixtures for the provided tests.

The provided tests do not know your class names. For each file under src/model/ they
use the LAST nn.Module subclass defined in that file (helpers first, main class last).
While a file is still a stub (no nn.Module defined), the tests that need it are skipped.
"""

import importlib
import inspect

import pytest
from torch import nn


def main_class(module_name):
    module = importlib.import_module(module_name)
    classes = [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, nn.Module) and cls.__module__ == module.__name__
    ]
    if not classes:
        pytest.skip(f"{module_name} does not define an nn.Module yet (still a stub)")
    return max(classes, key=lambda cls: inspect.getsourcelines(cls)[1])


@pytest.fixture(scope="session")
def attention_cls():
    return main_class("src.model.attention")


@pytest.fixture(scope="session")
def encoder_cls():
    return main_class("src.model.encoder")


@pytest.fixture(scope="session")
def vlm_cls():
    return main_class("src.model.vlm")
