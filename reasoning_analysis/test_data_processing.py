#!/usr/bin/env python3
"""
Unit tests for data_processing module
"""

import unittest
import numpy as np
from typing import List, Dict, Any

from data_processing import (
    compute_model_family_best_mse,
    compute_model_family_best_accuracy
)

class TestModelFamilyMetrics(unittest.TestCase):
    """Test case for model family metric computation functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample regression predictions
        self.regression_predictions = [
            {
                'features': [1.0, 2.0],
                'true_value': 1.5,
                'predicted': 1.4,
                'squared_error': 0.01
            },
            {
                'features': [2.0, 3.0],
                'true_value': 2.5,
                'predicted': 2.7,
                'squared_error': 0.04
            },
            {
                'features': [3.0, 4.0],
                'true_value': 3.5,
                'predicted': 3.4,
                'squared_error': 0.01
            },
            {
                'features': [4.0, 5.0],
                'true_value': 4.5,
                'predicted': 4.6,
                'squared_error': 0.01
            }
        ]
        
        # Create sample classification predictions
        self.classification_predictions = [
            {
                'features': [1.0, 2.0],
                'true_label': 0,
                'predicted': 0,
                'correct': True
            },
            {
                'features': [2.0, 3.0],
                'true_label': 1,
                'predicted': 1,
                'correct': True
            },
            {
                'features': [3.0, 4.0],
                'true_label': 0,
                'predicted': 0,
                'correct': True
            },
            {
                'features': [4.0, 5.0],
                'true_label': 1,
                'predicted': 1,
                'correct': True
            }
        ]
    
    def test_compute_model_family_best_mse_linear(self):
        """Test computing best MSE for linear regression."""
        mse = compute_model_family_best_mse(
            'sklearn.linear_model.LinearRegression',
            self.regression_predictions
        )
        self.assertIsNotNone(mse)
        self.assertIsInstance(mse, float)
        # For this simple linearly related data, MSE should be very low
        self.assertLess(mse, 0.1)
    
    def test_compute_model_family_best_mse_ridge(self):
        """Test computing best MSE for ridge regression."""
        mse = compute_model_family_best_mse(
            'sklearn.linear_model.Ridge',
            self.regression_predictions
        )
        self.assertIsNotNone(mse)
        self.assertIsInstance(mse, float)
        self.assertLess(mse, 0.1)
    
    def test_compute_model_family_best_mse_unknown(self):
        """Test computing best MSE for unknown model family."""
        mse = compute_model_family_best_mse(
            'unknown',
            self.regression_predictions
        )
        self.assertIsNone(mse)
    
    def test_compute_model_family_best_accuracy_logistic(self):
        """Test computing best accuracy for logistic regression."""
        accuracy = compute_model_family_best_accuracy(
            'sklearn.linear_model.LogisticRegression',
            self.classification_predictions
        )
        self.assertIsNotNone(accuracy)
        self.assertIsInstance(accuracy, float)
        # For this simple linearly separable data, accuracy should be high
        self.assertGreaterEqual(accuracy, 75.0)  # At least 75%
    
    def test_compute_model_family_best_accuracy_svm(self):
        """Test computing best accuracy for SVM."""
        accuracy = compute_model_family_best_accuracy(
            'sklearn.svm.SVC',
            self.classification_predictions
        )
        self.assertIsNotNone(accuracy)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 75.0)  # At least 75%
    
    def test_compute_model_family_best_accuracy_unknown(self):
        """Test computing best accuracy for unknown model family."""
        accuracy = compute_model_family_best_accuracy(
            'unknown',
            self.classification_predictions
        )
        self.assertIsNone(accuracy)
    
    def test_edge_cases(self):
        """Test edge cases like empty predictions."""
        # Empty predictions
        self.assertIsNone(compute_model_family_best_mse('linear', []))
        self.assertIsNone(compute_model_family_best_accuracy('logistic', []))
        
        # Missing features
        bad_predictions = [{'true_value': 1.5}]
        self.assertIsNone(compute_model_family_best_mse('linear', bad_predictions))
        
        # Too few samples
        single_prediction = [{'features': [1.0, 2.0], 'true_value': 1.5}]
        self.assertIsNone(compute_model_family_best_mse('linear', single_prediction))

if __name__ == '__main__':
    unittest.main() 