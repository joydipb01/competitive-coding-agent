import requests
import os
from dotenv import load_dotenv

load_dotenv()

JUDGE_ZERO_API_KEY = os.getenv("JUDGE_ZERO_API_KEY")

def check_correctness(code: str, input_data: str, expected_output: str, timeout: int = 5) -> bool:
    """
    Check the correctness of the code by sending a request to a C++ execution environment.
    """
    url = "https://judge0-ce.p.rapidapi.com/submissions?base64_encoded=false&wait=true&fields=*"

    headers = {
        "X-RapidAPI-Key": JUDGE_ZERO_API_KEY,
        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com"
    }

    payload = {
        "code": code,
        "input": input_data,
        "expected_output": expected_output
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        result = response.json()
        token = result.get("token")
        if token:
            status_url = f"https://judge0-ce.p.rapidapi.com/submissions/{token}?base64_encoded=false"
            status_response = requests.get(status_url, headers=headers, timeout=timeout)
            status_result = status_response.json()
            return status_result.get("status", {}).get("description") == "Accepted"
        else:
            print("No token received in response.")
            return False
    except requests.RequestException as e:
        print(f"Error during request: {e}")
        return False