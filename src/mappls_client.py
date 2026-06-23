import requests
import json
import logging
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MapplsClient:
    def __init__(self, api_key: str, cache_dir: str):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, api_name: str, params: dict) -> Path:
        # Create deterministic hash for cache key
        param_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.md5(param_str.encode()).hexdigest()
        return self.cache_dir / f"mappls_{api_name}_{hash_str}.json"
        
    def _cached_request(self, api_name: str, url: str, params: dict) -> dict:
        cache_path = self._get_cache_path(api_name, params)
        
        if cache_path.exists():
            logging.debug(f"Cache hit for {api_name}: {params}")
            with open(cache_path, 'r') as f:
                return json.load(f)
                
        logging.info(f"API call for {api_name}: {params}")
        
        # We don't want to log the api key, so we insert it here
        full_url = url.format(api_key=self.api_key)
        
        try:
            response = requests.get(full_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            return data
            
        except Exception as e:
            logging.error(f"Mappls API failed: {str(e)}")
            return {}

    def get_route(self, origin: tuple, dest: tuple) -> dict:
        """ origin and dest are (lat, lng) """
        orig_str = f"{origin[1]},{origin[0]}" # lng,lat
        dest_str = f"{dest[1]},{dest[0]}"
        
        url = "https://apis.mappls.com/advancedmaps/v1/{api_key}/route_adv/driving/" + f"{orig_str};{dest_str}"
        return self._cached_request('route', url, {})

    def get_route_avoiding(self, origin: tuple, dest: tuple, avoid: tuple) -> dict:
        orig_str = f"{origin[1]},{origin[0]}"
        dest_str = f"{dest[1]},{dest[0]}"
        avoid_str = f"point:{avoid[0]},{avoid[1]}" # avoid takes lat,lng
        
        url = "https://apis.mappls.com/advancedmaps/v1/{api_key}/route_adv/driving/" + f"{orig_str};{dest_str}"
        return self._cached_request('route_avoid', url, {'exclude': avoid_str})
