import requests


def get_weather(city: str) -> str:
    """Query current weather information from wttr.in."""
    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city} current weather: {weather_desc}, temperature {temp_c} degrees Celsius"
    except requests.exceptions.RequestException as error:
        return f"Error: Network problem encountered when querying weather - {error}"
    except (KeyError, IndexError) as error:
        return f"Error: Failed to parse weather data, city name may be invalid - {error}"

