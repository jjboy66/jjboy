import base64
import importlib.util
import io
import unittest


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE and PIL_AVAILABLE, "fastapi/pillow not installed")
class ApiTest(unittest.TestCase):
    def test_detect_endpoint(self):
        from fastapi.testclient import TestClient
        from PIL import Image

        from src.api.app import app

        img = Image.new("RGB", (120, 80), color=(255, 255, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        client = TestClient(app)
        response = client.post(
            "/detect",
            json={"image_base64": image_b64, "source_id": "test-cam", "frame_index": 1},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("detections", payload)


if __name__ == "__main__":
    unittest.main()
