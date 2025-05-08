import pytest
from src.services.funding_rates import fetch_funding_rates, analyze_funding_rates

class MockAPI:
    def get_funding_rates(self, symbol):
        if symbol == 'BTCUSDT':
            return [
                {'timestamp': 1700000000, 'rate': 0.0001},
                {'timestamp': 1700003600, 'rate': 0.0002},
                {'timestamp': 1700007200, 'rate': 0.00015},
            ]
        elif symbol == 'FAIL':
            raise Exception('API error')
        else:
            return []

@pytest.fixture
def mock_api():
    return MockAPI()

def test_fetch_funding_rates_success(mock_api):
    rates = fetch_funding_rates('BTCUSDT', api=mock_api)
    assert isinstance(rates, list)
    assert all('rate' in r for r in rates)
    assert len(rates) == 3

def test_fetch_funding_rates_failure(mock_api):
    with pytest.raises(Exception):
        fetch_funding_rates('FAIL', api=mock_api)

def test_analyze_funding_rates_basic(mock_api):
    rates = fetch_funding_rates('BTCUSDT', api=mock_api)
    analysis = analyze_funding_rates(rates)
    assert 'current' in analysis
    assert 'average' in analysis
    assert 'history' in analysis
    assert abs(analysis['average'] - 0.00015) < 1e-6

def test_analyze_funding_rates_empty():
    analysis = analyze_funding_rates([])
    assert analysis['current'] is None
    assert analysis['average'] is None
    assert analysis['history'] == []

# CLI integration and educational output tests would be added in integration tests or CLI test suite 