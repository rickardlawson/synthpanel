"""Enkel innlogging (HTTP Basic) for demo-miljøet.

Slås på når SYNTHPANEL_PASSWORD er satt (brukernavn: SYNTHPANEL_USER, standard
«demo»). Uten passord er tjenesten åpen, som lokalt. /health er alltid åpen så
driftsplattformen kan sjekke at tjenesten lever.
"""
from __future__ import annotations

import base64
import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

OPEN_PATHS = {"/health"}


def _credentials_ok(header: str | None, user: str, password: str) -> bool:
    if not header or not header.startswith("Basic "):
        return False
    try:
        given_user, _, given_pw = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except Exception:
        return False
    return secrets.compare_digest(given_user, user) & secrets.compare_digest(given_pw, password)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        password = os.environ.get("SYNTHPANEL_PASSWORD")
        if not password or request.url.path in OPEN_PATHS:
            return await call_next(request)
        user = os.environ.get("SYNTHPANEL_USER", "demo")
        if _credentials_ok(request.headers.get("authorization"), user, password):
            return await call_next(request)
        return Response("Innlogging kreves", status_code=401,
                        headers={"WWW-Authenticate": 'Basic realm="Synthpanel", charset="UTF-8"'})
