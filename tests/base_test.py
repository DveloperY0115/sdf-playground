"""
base_test.py

A collection of utilities used for testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Type

import torch

from src.configs.base_config import InstantiateConfig


@dataclass
class TestConfig(InstantiateConfig):
    """The configuration of a test"""

    _target: Type = field(default_factory=lambda: Test)
    output_root: Path = Path("output/test")
    """The root directory of outputs"""
    test_name: str = "base_test"
    """The name of the test"""
    test_cases: List[Callable] = field(default_factory=lambda: [])
    """The list of test cases to run"""
    device: torch.device = torch.device("cuda")
    """The device to run the test on"""

class Test:
    """The base class """

    config: TestConfig
    test_cases: List[Callable]

    def __init__(self, config: TestConfig) -> None:
        self.config = config

        self._register_test_cases()
        self.device = self.config.device

    def _register_test_cases(self) -> None:
        """Registers the test cases specified in the config"""
        self.test_cases = self.config.test_cases

    def run(self) -> None:
        """Runs test(s)"""

        # create directory for the current experiment
        exp_dir = self.config.output_root / self.config.test_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # run each test case
        for test_case in self.test_cases:
            test_dir = exp_dir / test_case.__name__  
            test_dir.mkdir(parents=True, exist_ok=True)
            test_case(test_dir, self.device)

        print(
            f"Test [{self.__class__.__name__}] successfully passed all tests."
        )
