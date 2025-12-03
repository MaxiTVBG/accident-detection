import sys
import os
from unittest.mock import patch, MagicMock

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import get_gps_coordinates, get_gemini_api_key

def test_get_gps_coordinates_manual():
    """
    Tests if manual coordinates are returned correctly.
    """
    lat, lon = 34.0522, -118.2437
    assert get_gps_coordinates(manual_coords=(lat, lon)) == (lat, lon)

@patch('src.utils.geocoder.osm')
def test_get_gps_coordinates_by_location_success(mock_osm):
    """
    Tests successful geocoding by location name.
    """
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.latlng = [34.0522, -118.2437]
    mock_osm.return_value = mock_response

    assert get_gps_coordinates(location="Los Angeles") == [34.0522, -118.2437]
    mock_osm.assert_called_once_with("Los Angeles")

@patch('src.utils.geocoder.osm')
@patch('src.utils.geocoder.ip')
def test_get_gps_coordinates_by_location_fail_fallback_to_ip(mock_ip, mock_osm):
    """
    Tests fallback to IP geocoding when location geocoding fails.
    """
    mock_osm.return_value.ok = False
    
    mock_ip_response = MagicMock()
    mock_ip_response.ok = True
    mock_ip_response.latlng = [40.7128, -74.0060]
    mock_ip.return_value = mock_ip_response

    assert get_gps_coordinates(location="Invalid Location") == [40.7128, -74.0060]
    mock_ip.assert_called_once_with('me')

@patch('src.utils.geocoder.osm')
@patch('src.utils.geocoder.ip')
def test_get_gps_coordinates_all_fail_fallback_to_default(mock_ip, mock_osm):
    """
    Tests fallback to default coordinates when all geocoding fails.
    """
    mock_osm.return_value.ok = False
    mock_ip.return_value.ok = False

    # Note the default coordinates from the function
    assert get_gps_coordinates(location="Invalid Location") == (37.3349, -122.0091)

@patch('src.utils.load_dotenv')
def test_get_gemini_api_key_not_set(mock_load_dotenv, monkeypatch):
    """
    Tests that None is returned when the API key is not in the environment.
    """
    # Use monkeypatch to temporarily delete the environment variable
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_gemini_api_key() is None

@patch('src.utils.load_dotenv')
def test_get_gemini_api_key_is_set(mock_load_dotenv, monkeypatch):
    """
    Tests that the API key is returned correctly when it is in the environment.
    """
    api_key = "test_api_key_12345"
    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    assert get_gemini_api_key() == api_key
