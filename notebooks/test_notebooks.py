#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions to test the Pandora notebooks.
"""
import subprocess
import unittest


class TestPandora(unittest.TestCase):
    """
    TestPandora class allows to test the pandora notebooks
    """

    @staticmethod
    def test_analyse_demo():
        """
        Test that the Analyse_demo notebook runs without errors

        """
        subprocess.run(['jupyter nbconvert --to script notebooks/Analyse_demo.ipynb --output-dir notebooks'],
                           shell=True, check=False)

        out = subprocess.run(['ipython Analyse_demo.py'],
                                 shell=True, check=False,
                                 cwd='notebooks',
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

        assert out.returncode == 0

    @staticmethod
    def test_usage_demo_multiscale():
        """
        Test that the Usage_demo_multiscale notebook runs without errors

        """
        subprocess.run(['jupyter nbconvert --to script notebooks/Usage_demo_multiscale.ipynb --output-dir notebooks'],
                           shell=True, check=False)

        out = subprocess.run(['ipython Usage_demo_multiscale.py'],
                                 shell=True, check=False,
                                 cwd='notebooks',
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

        assert out.returncode == 0

    @staticmethod
    def test_usage_demo():
        """
        Test that the Usage_demo notebook runs without errors

        """
        subprocess.run(['jupyter nbconvert --to script notebooks/Usage_demo.ipynb --output-dir notebooks'],
                           shell=True, check=False)
        out = subprocess.run(['ipython Usage_demo.py'],
                                 shell=True, check=False,
                                 cwd='notebooks',
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

        assert out.returncode == 0
