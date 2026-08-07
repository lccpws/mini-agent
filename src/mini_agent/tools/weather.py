from mini_agent.tools.base_tool import BaseTool
import requests
import time


class WeatherTool(BaseTool):
    name = "weather"
    description = "查询天气"
    capabilities = ["weather"]

    version = "1.0.0"
    author = "MiniAgent"
    allow_roles = ["admin", "user"]
    required_permissions = ["network"]
    risk_level = 1

    MAX_RETRIES = 3
    RETRY_DELAY = 1

    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如：北京、上海、广州"
            }
        },
        "required": ["city"]
    }

    def execute(self, city):
        if not city:
            return {"error": "请输入城市名称"}

        city_info = self._request_with_retry(self.get_coordinates, city)
        if not city_info:
            return {"error": f"未找到城市: {city}"}

        weather_data = self._request_with_retry(
            self.get_weather, city_info["latitude"], city_info["longitude"], city_info["timezone"]
        )
        if not weather_data:
            return {"error": "无法获取天气信息"}

        return self._format_weather_dict(city_info, weather_data)

    def _request_with_retry(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                last_error = f"返回结果为空"
            except Exception as e:
                last_error = str(e)

            if attempt < self.MAX_RETRIES - 1:
                time.sleep(self.RETRY_DELAY * (attempt + 1))

        return None

    def get_coordinates(self, city_name: str) -> dict | None:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city_name,
            "count": 1,
            "language": "zh",
            "format": "json",
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("results"):
            return None

        result = data["results"][0]
        return {
            "name": result.get("name", city_name),
            "country": result.get("country", ""),
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "timezone": result.get("timezone", "Asia/Shanghai"),
        }

    def get_weather(self, latitude: float, longitude: float, timezone: str = "Asia/Shanghai") -> dict | None:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "forecast_days": 3,
            "timezone": timezone,
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def decode_weather_code(self, code: int) -> str:
        weather_map = {
            0: "晴天",
            1: "大部晴朗",
            2: "局部多云",
            3: "多云",
            45: "雾",
            48: "冻雾",
            51: "小毛毛雨",
            53: "中毛毛雨",
            55: "大毛毛雨",
            56: "冻毛毛雨（轻）",
            57: "冻毛毛雨（重）",
            61: "小雨",
            63: "中雨",
            65: "大雨",
            66: "冻雨（轻）",
            67: "冻雨（重）",
            71: "小雪",
            73: "中雪",
            75: "大雪",
            77: "雪粒",
            80: "阵雨（小）",
            81: "阵雨（中）",
            82: "阵雨（大）",
            85: "阵雪（小）",
            86: "阵雪（大）",
            95: "雷暴",
            96: "雷暴伴冰雹（轻）",
            99: "雷暴伴冰雹（重）",
        }
        return weather_map.get(code, f"未知({code})")

    def _format_weather_dict(self, city_info: dict, weather_data: dict) -> dict:
        current = weather_data["current"]
        daily = weather_data["daily"]

        forecast = []
        for i in range(len(daily["time"])):
            forecast.append({
                "date": daily["time"][i],
                "weather": self.decode_weather_code(daily["weather_code"][i]),
                "temp_max": daily["temperature_2m_max"][i],
                "temp_min": daily["temperature_2m_min"][i],
                "precipitation": daily["precipitation_sum"][i],
                "wind_max": daily["wind_speed_10m_max"][i],
            })

        return {
            "city": city_info["name"],
            "country": city_info["country"],
            "temperature": current["temperature_2m"],
            "apparent_temperature": current["apparent_temperature"],
            "weather": self.decode_weather_code(current["weather_code"]),
            "humidity": current["relative_humidity_2m"],
            "wind_speed": current["wind_speed_10m"],
            "forecast": forecast,
        }

    def format_weather_report(self, city_info: dict, weather_data: dict) -> str:
        data = self._format_weather_dict(city_info, weather_data)
        lines = [
            f" {data['city']}（{data['country']}）",
            f"当前温度：{data['temperature']}°C",
            f"体感温度：{data['apparent_temperature']}°C",
            f"湿度：{data['humidity']}%",
            f"风速：{data['wind_speed']} km/h",
            f"天气：{data['weather']}",
            "",
            "未来3天预报：",
        ]
        for f in data["forecast"]:
            lines.append(
                f"  {f['date']}  {f['weather']}  "
                f"{f['temp_min']}~{f['temp_max']}°C  "
                f"降水{f['precipitation']}mm  "
                f"最大风速{f['wind_max']}km/h"
            )
        return "\n".join(lines)
