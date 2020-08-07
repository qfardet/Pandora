#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of PANDORA
#
#     https://github.com/CNES/Pandora_pandora
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions to test the margin module.
"""

import numpy as np
import pandora.marge
import unittest


class TestMargins(unittest.TestCase):
    """
    TestMargins class allows to test the marge module
    """
    def setUp(self):
        pass

    def test_get_margins(self):
        """
        Test get_margins function
        """
        # Test with SGM configuration
        cfg_sgm = {'stereo': {'stereo_method': 'census', 'window_size': 5}, 'optimization': {'optimization_method': 'sgm'},
                   'refinement': {'refinement_method': 'vfit'}, 'filter': {'filter_method': 'median', 'filter_size': 3},
                   'validation': {'validation_method': 'cross_checking', 'cross_checking_threshold': 1,
                                  'right_left_mode': 'accurate'}}

        res = pandora.marge.get_margins(-13, 14, cfg_sgm)
        np.testing.assert_array_equal(res.loc[dict(image='ref_margin')].values, np.array([54, 40, 54, 40]))
        np.testing.assert_array_equal(res.loc[dict(image='sec_margin')].values, np.array([54, 40, 54, 40]))
        assert res.attrs['disp_min'] == -13
        assert res.attrs['disp_max'] == 14

        res = pandora.marge.get_margins(3, 14, cfg_sgm)
        np.testing.assert_array_equal(res.loc[dict(image='ref_margin')].values, np.array([54, 40, 54, 40]))
        np.testing.assert_array_equal(res.loc[dict(image='sec_margin')].values, np.array([54, 40, 54, 40]))
        assert res.attrs['disp_min'] == 3
        assert res.attrs['disp_max'] == 14

        res = pandora.marge.get_margins(-13, -2, cfg_sgm)
        np.testing.assert_allclose(res.loc[dict(image='ref_margin')].values,
                                   np.array([53, 40, 53, 40]))
        np.testing.assert_allclose(res.loc[dict(image='sec_margin')].values,
                                   np.array([53, 40, 53, 40]))
        assert res.attrs['disp_min'] == -13
        assert res.attrs['disp_max'] == -2

        # Test without SGM configuration
        cfg = {'stereo': {'stereo_method': 'census', 'window_size': 3}, 'optimization': {'optimization_method': 'none'},
               'refinement': {'refinement_method': 'vfit'}, 'filter': {'filter_method': 'median', 'filter_size': 3},
               'validation': {'validation_method': 'cross_checking', 'cross_checking_threshold': 1,
                              'right_left_mode': 'accurate'}}

        res = pandora.marge.get_margins(-13, 14, cfg)
        np.testing.assert_array_equal(res.loc[dict(image='ref_margin')].values, np.array([17, 2, 17, 2]))
        np.testing.assert_array_equal(res.loc[dict(image='sec_margin')].values, np.array([17, 2, 17, 2]))
        assert res.attrs['disp_min'] == -13
        assert res.attrs['disp_max'] == 14

        cfg = {'stereo': {'stereo_method': 'sad', 'window_size': 9}, 'optimization': {'optimization_method': 'none'},
               'refinement': {'refinement_method': 'vfit'}, 'filter': {'filter_method': 'median', 'filter_size': 3},
               'validation': {'validation_method': 'cross_checking', 'cross_checking_threshold': 1,
                              'right_left_mode': 'accurate'}}

        res = pandora.marge.get_margins(3, 14, cfg)
        np.testing.assert_array_equal(res.loc[dict(image='ref_margin')].values, np.array([20, 5, 20, 5]))
        np.testing.assert_array_equal(res.loc[dict(image='sec_margin')].values, np.array([20, 5, 20, 5]))
        assert res.attrs['disp_min'] == 3
        assert res.attrs['disp_max'] == 14

        cfg = {'stereo': {'stereo_method': 'sad', 'window_size': 1}, 'optimization': {'optimization_method': 'none'},
               'refinement': {'refinement_method': 'vfit'}, 'filter': {'filter_method': 'median', 'filter_size': 5},
               'validation': {'validation_method': 'cross_checking', 'cross_checking_threshold': 1,
                              'right_left_mode': 'accurate'}}

        res = pandora.marge.get_margins(-13, -2, cfg)
        np.testing.assert_array_equal(res.loc[dict(image='ref_margin')].values, np.array([16, 2, 16, 2]))
        np.testing.assert_array_equal(res.loc[dict(image='sec_margin')].values, np.array([16, 2, 16, 2]))
        assert res.attrs['disp_min'] == -13
        assert res.attrs['disp_max'] == -2


if __name__ == '__main__':
    unittest.main()