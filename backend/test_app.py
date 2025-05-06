import pytest
import requests

BASE_URL = "http://localhost:5000"

#Registration
def test_registration_success():
    payload = {
        "username": "pytest_user",
        "confirmUsername": "pytest_user",
        "email": "pytest@example.com",
        "password": "pytest123",
        "confirmPassword": "pytest123"
    }
    res = requests.post(f"{BASE_URL}/register", json=payload)
    assert res.status_code in (200, 400)  # 400 if user already exists

def test_registration_fail_mismatch_password():
    payload = {
        "username": "user2",
        "confirmUsername": "user2",
        "email": "user2@example.com",
        "password": "pass1",
        "confirmPassword": "pass2"
    }
    res = requests.post(f"{BASE_URL}/register", json=payload)
    assert res.status_code == 400
    assert "do not match" in res.text

#Login
def test_login_success():
    payload = {
        "username": "pytest_user",
        "password": "pytest123"
    }
    res = requests.post(f"{BASE_URL}/login", json=payload)
    assert res.status_code == 200
    assert "Login successful" in res.text

def test_login_invalid_credentials():
    payload = {
        "username": "wronguser",
        "password": "wrongpass"
    }
    res = requests.post(f"{BASE_URL}/login", json=payload)
    assert res.status_code == 401

#Predict 
def test_predict_missing_image():
    res = requests.post(f"{BASE_URL}/predict", files={})
    assert res.status_code == 400
    assert "required" in res.text
