# Phase 23.1 — Authentication & Access Control

## Overview

Phase 23.1 adds **user authentication** (login/register) before accessing the unified Streamlit application and optional JWT protection for Phase 20/21 API routes.

This is **additive** and separate from the existing Phase 13 **API-key** middleware.

## Architecture

```
Streamlit app.py
  └─ Auth gate (session)
       └─ SQLite users DB
       └─ PBKDF2 password hashes
       └─ Signed JWT tokens (itsdangerous)

FastAPI
  ├─ POST /auth/register  (public)
  ├─ POST /auth/login     (public)
  ├─ GET  /auth/me        (Bearer token)
  ├─ /phase20/*           (JWT when FORESIGHT_USER_AUTH_REQUIRED=true)
  └─ /phase21/*           (ADMIN JWT when required)
```

## Database

- Path: `data/auth/project_foresight_auth.db` (override: `FORESIGHT_AUTH_DB_PATH`)
- Separate from ML datasets and model files
- Fields: id, full_name, email, password_hash, is_active, role, created_at, last_login

## Password Security

- **Never stored in plain text**
- PBKDF2-HMAC-SHA256 (260,000 iterations)
- Policy: minimum 8 characters, at least one letter and one number
- Passwords and hashes are never returned in API responses

## Registration Flow

1. User submits name, email, password, confirm password
2. Validation: unique email, matching passwords, policy check
3. Account created with role `USER`
4. Response: success message only (no secrets)

## Login Flow

1. User submits email + password
2. Service verifies user exists, is active, password valid
3. Returns Bearer `access_token` + public user profile
4. Streamlit stores session state and shows sidebar navigation

## Session Management

- **Streamlit:** `st.session_state` (authenticated, user profile, token)
- **API:** `Authorization: Bearer <token>` header
- Tokens signed with `JWT_SECRET_KEY` or `SECRET_KEY`
- Default expiry: 86400 seconds (24h)

## Protected Routes

| Area | Protection |
|------|------------|
| Unified app (`app.py`) | Always requires login |
| `/phase20/*` | JWT when `FORESIGHT_USER_AUTH_REQUIRED=true` |
| `/phase21/*` | ADMIN JWT when required |
| `/health`, `/ready` | Public |
| `/auth/register`, `/auth/login` | Public |

Set `FORESIGHT_USER_AUTH_REQUIRED=true` in production-style API deployments.

## Role-Based Access

| Role | Access |
|------|--------|
| USER | Dashboards, forecasting, inventory, analytics, model info, documentation, about |
| ADMIN | All USER pages + monitoring, alerts, integrity, validation status |

**Assign ADMIN** (explicit database update only):

```sql
UPDATE users SET role = 'ADMIN' WHERE email = 'admin@example.com';
```

Do not create fake admin users in code.

## Logout

- Sidebar **Logout** clears session state
- Client should discard Bearer token
- Protected pages inaccessible after logout

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `JWT_SECRET_KEY` / `SECRET_KEY` | Token signing (required in production) |
| `FORESIGHT_AUTH_DB_PATH` | SQLite path |
| `FORESIGHT_USER_AUTH_REQUIRED` | Enforce JWT on Phase 20/21 API |
| `FORESIGHT_JWT_EXPIRY_SECONDS` | Token lifetime |

Copy `.env.example` — never commit `.env`.

## Security Limitations

- Local SQLite — not a production identity provider
- No email verification or password reset flow
- No OAuth/SSO
- API-key auth (Phase 13) remains separate
- Development default secret must be changed for production

## Unchanged Systems

Authentication does **not** modify:

- Model weights or `models/final/`
- Phase 20 forecasting logic
- Phase 21 monitoring calculations
- Risk engine
