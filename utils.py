import logging
import geocoder
import os
from dotenv import load_dotenv

def setup_logging():
    """Configures the logging for the application."""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger(__name__)

def get_gps_coordinates(manual_coords=None, location=None):
    """
    Fetches GPS coordinates.
    
    Priority is given to manual coordinates, then location-based geocoding,
    and finally falls back to IP-based geocoding with a warning.
    
    Args:
        manual_coords (tuple, optional): Manual latitude and longitude. Defaults to None.
        location (str, optional): A location name or address to geocode. Defaults to None.

    Returns:
        tuple: Latitude and longitude.
    """
    if manual_coords:
        return manual_coords
    
    if location:
        try:
            g = geocoder.osm(location)
            if g.ok:
                logging.info(f"Successfully geocoded '{location}' to {g.latlng}")
                return g.latlng
            else:
                logging.warning(f"Could not geocode location '{location}'.")
        except Exception as e:
            logging.error(f"An error occurred during geocoding: {e}")

    logging.warning("No location provided or geocoding failed. Falling back to IP-based location, which can be inaccurate.")
    try:
        g = geocoder.ip('me')
        if g.ok:
            return g.latlng
        else:
            logging.warning("Could not get location from IP. Returning default coordinates.")
            return 37.3349, -122.0091
    except Exception as e:
        logging.error(f"An error occurred while fetching GPS coordinates from IP: {e}")
        return 37.3349, -122.0091

def get_gemini_api_key():
    """
    Loads the Gemini API key from environment variables.
    
    Returns:
        str or None: The API key if found, otherwise None.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.warning("GEMINI_API_KEY is not set. Accident descriptions will not be generated.")
        logging.warning("Get a FREE API key at: https://makersuite.google.com/app/apikey")
    return api_key
