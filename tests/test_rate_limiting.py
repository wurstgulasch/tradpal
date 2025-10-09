#!/usr/bin/env python3
"""
Tests for Adaptive Rate Limiting Module
Tests intelligent API rate limiting with exchange-specific limits.
"""

import pytest
import time
from unittest.mock import patch, MagicMock
from src.data_fetcher import AdaptiveRateLimiter


class TestAdaptiveRateLimiter:
    """Test suite for adaptive rate limiting functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization with default settings."""
        limiter = AdaptiveRateLimiter()
        assert limiter.exchange_name == 'kraken'  # Default exchange
        assert isinstance(limiter.request_times, list)
        assert isinstance(limiter.rate_limits, dict)
        assert 'requests_per_second' in limiter.rate_limits
        assert 'requests_per_minute' in limiter.rate_limits

    def test_rate_limiter_custom_exchange(self):
        """Test rate limiter with custom exchange."""
        limiter = AdaptiveRateLimiter('binance')
        assert limiter.exchange_name == 'binance'
        # Should have different limits for binance
        assert limiter.rate_limits['requests_per_second'] == 10

    @patch('time.time')
    def test_can_make_request_no_limits(self, mock_time):
        """Test request permission when no limits exceeded."""
        mock_time.return_value = 1000.0
        limiter = AdaptiveRateLimiter()

        # No requests recorded yet
        assert limiter.can_make_request() is True

    @patch('time.time')
    def test_can_make_request_under_limit(self, mock_time):
        """Test request permission when under rate limits."""
        mock_time.return_value = 1000.0
        limiter = AdaptiveRateLimiter()

        # Add some requests but stay under limits
        for i in range(5):
            limiter.record_request()
            mock_time.return_value += 10  # Space out requests

        assert limiter.can_make_request() is True

    @patch('time.time')
    def test_can_make_request_rate_limit_exceeded(self, mock_time):
        """Test request blocking when rate limit exceeded."""
        mock_time.return_value = 1000.0
        limiter = AdaptiveRateLimiter()

        # Add many requests in short time to exceed limits
        for i in range(25):  # Exceed per-minute limit
            limiter.record_request()

        # Should be blocked
        assert limiter.can_make_request() is False

    @patch('time.time')
    def test_request_cleanup(self, mock_time):
        """Test cleanup of old requests."""
        limiter = AdaptiveRateLimiter()
        base_time = 1000.0

        # Add requests over time
        for i in range(10):
            mock_time.return_value = base_time + i * 10
            limiter.record_request()

        # Move time forward past cleanup window
        mock_time.return_value = base_time + 120  # 2 minutes later

        # Next check should trigger cleanup
        result = limiter.can_make_request()

        # Should have cleaned up old requests
        assert len(limiter.request_times) < 10

    def test_get_backoff_time(self):
        """Test backoff time calculation."""
        limiter = AdaptiveRateLimiter()

        # Test increasing backoff times
        backoff1 = limiter.get_backoff_time(1)
        backoff2 = limiter.get_backoff_time(2)
        backoff3 = limiter.get_backoff_time(3)

        assert backoff2 > backoff1
        assert backoff3 > backoff2
        assert backoff3 <= 300  # Should be capped

    def test_get_backoff_time_with_errors(self):
        """Test backoff time increases with consecutive errors."""
        limiter = AdaptiveRateLimiter()

        # Record consecutive errors
        limiter.record_error()
        limiter.record_error()
        limiter.record_error()

        backoff = limiter.get_backoff_time(1)
        # Should be doubled due to consecutive errors
        normal_backoff = min(2 ** 1, 300)
        assert backoff >= normal_backoff * 2

    def test_error_tracking(self):
        """Test error and success tracking."""
        limiter = AdaptiveRateLimiter()

        # Record errors
        limiter.record_error()
        assert limiter.consecutive_errors == 1

        limiter.record_error()
        assert limiter.consecutive_errors == 2

        # Record success
        limiter.record_success()
        assert limiter.consecutive_errors == 1

        # Record more successes
        limiter.record_success()
        limiter.record_success()
        assert limiter.consecutive_errors == 0

    def test_exchange_specific_limits(self):
        """Test different rate limits for different exchanges."""
        kraken_limiter = AdaptiveRateLimiter('kraken')
        binance_limiter = AdaptiveRateLimiter('binance')
        coinbase_limiter = AdaptiveRateLimiter('coinbase')
        unknown_limiter = AdaptiveRateLimiter('unknown')

        # Each should have appropriate limits
        assert kraken_limiter.rate_limits['requests_per_minute'] == 20
        assert binance_limiter.rate_limits['requests_per_minute'] == 1200
        assert coinbase_limiter.rate_limits['requests_per_minute'] == 100
        assert unknown_limiter.rate_limits['requests_per_minute'] == 60  # Default

    @patch('time.sleep')
    @patch('src.data_fetcher.AdaptiveRateLimiter.can_make_request')
    @patch('src.data_fetcher.AdaptiveRateLimiter.record_request')
    def test_adaptive_retry_decorator_success(self, mock_record, mock_can_make, mock_sleep):
        """Test adaptive retry decorator on successful call."""
        from src.data_fetcher import adaptive_retry

        mock_can_make.return_value = True

        @adaptive_retry(max_retries=3)
        def test_function():
            return "success"

        result = test_function()
        assert result == "success"
        mock_record.assert_called_once()
        mock_sleep.assert_not_called()

    @patch('time.sleep')
    @patch('src.data_fetcher.AdaptiveRateLimiter.can_make_request')
    @patch('src.data_fetcher.AdaptiveRateLimiter.record_request')
    @patch('src.data_fetcher.AdaptiveRateLimiter.record_error')
    def test_adaptive_retry_decorator_with_retry(self, mock_record_error, mock_record, mock_can_make, mock_sleep):
        """Test adaptive retry decorator with retries."""
        from src.data_fetcher import adaptive_retry

        call_count = 0

        @adaptive_retry(max_retries=2)
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        mock_can_make.return_value = True

        result = test_function()
        assert result == "success"
        assert call_count == 3
        assert mock_record.call_count == 3
        assert mock_record_error.call_count == 2
        assert mock_sleep.call_count == 2