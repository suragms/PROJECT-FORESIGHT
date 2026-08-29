"""Phase 23.1 — Authentication tests."""

import hashlib
import json
import os

import pytest
from fastapi.testclient import TestClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(BASE, "docs")


@pytest.fixture()
def auth_db(tmp_path, monkeypatch):
    db = tmp_path / "test_auth.db"
    monkeypatch.setenv("FORESIGHT_AUTH_DB_PATH", str(db))
    from src.auth import service as auth_service_mod
    auth_service_mod._service = None
    return db


@pytest.fixture()
def auth_service(auth_db):
    from src.auth.service import AuthService
    return AuthService(auth_db)


@pytest.fixture()
def api_client(auth_db, monkeypatch):
    monkeypatch.setenv("FORESIGHT_USER_AUTH_REQUIRED", "true")
    from src.auth import service as auth_service_mod
    auth_service_mod._service = None
    from src.api.app import create_app
    return TestClient(create_app())


def _register(client, email="user@example.com", password="Secret123"):
    return client.post(
        "/auth/register",
        json={
            "full_name": "Test User",
            "email": email,
            "password": password,
            "confirm_password": password,
        },
    )


def _login(client, email="user@example.com", password="Secret123"):
    return client.post("/auth/login", json={"email": email, "password": password})


class TestRegistration:
    def test_register_success(self, auth_service):
        user = auth_service.register("Alice", "alice@test.com", "Secret123", "Secret123")
        assert user.email == "alice@test.com"
        assert user.role == "USER"

    def test_duplicate_email(self, auth_service):
        auth_service.register("Alice", "dup@test.com", "Secret123", "Secret123")
        from src.auth.service import AuthError
        with pytest.raises(AuthError):
            auth_service.register("Bob", "dup@test.com", "Secret123", "Secret123")

    def test_password_not_in_response(self, api_client):
        r = _register(api_client)
        assert r.status_code == 200
        assert "password_hash" not in r.text


class TestPasswordSecurity:
    def test_hash_and_verify(self):
        from src.auth.security import hash_password, verify_password
        hashed = hash_password("Secret123")
        assert "pbkdf2_sha256" in hashed
        assert verify_password("Secret123", hashed)
        assert not verify_password("wrong", hashed)

    def test_weak_password_rejected(self):
        from src.auth.security import validate_password_policy
        with pytest.raises(ValueError):
            validate_password_policy("short")


class TestLogin:
    def test_login_success(self, auth_service):
        auth_service.register("Bob", "bob@test.com", "Secret123", "Secret123")
        user, token = auth_service.login("bob@test.com", "Secret123")
        assert user.full_name == "Bob"
        assert token

    def test_invalid_login(self, auth_service):
        auth_service.register("Bob", "bob@test.com", "Secret123", "Secret123")
        from src.auth.service import AuthError
        with pytest.raises(AuthError):
            auth_service.login("bob@test.com", "WrongPass1")


class TestAPIProtection:
    def test_phase20_unauthorized_when_required(self, api_client):
        r = api_client.get("/phase20/model")
        assert r.status_code == 401

    def test_phase20_with_token(self, api_client):
        _register(api_client)
        login = _login(api_client)
        token = login.json()["access_token"]
        r = api_client.get("/phase20/model", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200

    def test_phase21_forbidden_for_user(self, api_client):
        _register(api_client)
        login = _login(api_client)
        token = login.json()["access_token"]
        r = api_client.get("/phase21/health", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_phase21_admin_access(self, api_client, auth_db):
        from src.auth.service import AuthService
        svc = AuthService(auth_db)
        svc.register("Admin", "admin@test.com", "Secret123", "Secret123")
        import sqlite3
        with sqlite3.connect(auth_db) as conn:
            conn.execute("UPDATE users SET role='ADMIN' WHERE email='admin@test.com'")
            conn.commit()
        login = _login(api_client, "admin@test.com", "Secret123")
        token = login.json()["access_token"]
        r = api_client.get("/phase21/health", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200


class TestSessionAndUI:
    def test_auth_modules_import(self):
        import dashboard.components.auth_ui as auth_ui
        import dashboard.session_auth as session_auth
        assert hasattr(auth_ui, "render_auth_screen")
        assert hasattr(session_auth, "logout_user")

    def test_admin_only_pages_open_to_all_authenticated(self):
        from dashboard.navigation import ADMIN_ONLY_PAGES
        assert len(ADMIN_ONLY_PAGES) == 0

    def test_page_allowed_for_all_roles(self):
        from dashboard.session_auth import page_allowed
        assert page_allowed("home", "USER")
        assert page_allowed("alerts", "USER")
        assert page_allowed("system_health", "USER")
        assert page_allowed("alerts", "ADMIN")


class TestIntegrity:
    def test_frozen_models_unchanged(self):
        reg = json.load(open(os.path.join(DOCS, "final_model_registry.json")))
        for e in reg:
            mf = os.path.join(BASE, e["model_file"].replace("\\", os.sep))
            h = hashlib.sha256()
            with open(mf, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            assert h.hexdigest() == e["hash"]

    def test_phase20_unchanged(self):
        reg = json.load(open(os.path.join(DOCS, "phase20_production_registry.json")))
        p20 = os.path.join(BASE, "models", "final", "phase20", "phase20_synthetic_lightgbm.joblib")
        h = hashlib.sha256()
        with open(p20, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == reg[0]["hash"]


class TestDocumentation:
    def test_auth_doc_exists(self):
        assert os.path.exists(os.path.join(DOCS, "phase23_1_authentication.md"))


class TestProductionJwtFallback:
    def test_login_works_without_jwt_env_in_production(self, auth_db, monkeypatch):
        monkeypatch.setenv("FORESIGHT_ENV", "production")
        for key in ("JWT_SECRET_KEY", "FORESIGHT_API_JWT_SECRET", "FORESIGHT_JWT_SECRET_KEY", "SECRET_KEY"):
            monkeypatch.delenv(key, raising=False)
        from src.auth import service as auth_service_mod
        auth_service_mod._service = None
        from src.api.app import create_app
        client = TestClient(create_app())
        assert _register(client, "produser@example.com", "Secret123").status_code == 200
        login = _login(client, "produser@example.com", "Secret123")
        assert login.status_code == 200, login.text
        assert login.json()["access_token"]

