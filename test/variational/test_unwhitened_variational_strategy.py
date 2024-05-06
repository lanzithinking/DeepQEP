#!/usr/bin/env python3

import unittest

import torch

import sys
sys.path.insert(0,'../..')
import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase

POWER = 1.0

class TestUnwhitenedVariational(VariationalTestCase, unittest.TestCase):
    _power = POWER
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.UnwhitenedVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        if self.distribution_cls == gpytorch.variational.CholeskyVariationalDistribution:
            self.assertEqual(cholesky_mock.call_count, 3)  # One for each forward pass, once for initialization
        else:
            self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!

    def test_fantasy_call(self, *args, **kwargs):
        # we only want to check CholeskyVariationalDistribution
        if self.distribution_cls is gpytorch.variational.CholeskyVariationalDistribution:
            return super().test_fantasy_call(*args, **kwargs)

        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestUnwhitenedPredictive(TestUnwhitenedVariational):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestUnwhitenedRobust(TestUnwhitenedVariational):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestUnwhitenedMeanFieldVariational(TestUnwhitenedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldPredictive(TestUnwhitenedPredictive):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldRobust(TestUnwhitenedRobust):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedDeltaVariational(TestUnwhitenedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaPredictive(TestUnwhitenedPredictive):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaRobust(TestUnwhitenedRobust):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
