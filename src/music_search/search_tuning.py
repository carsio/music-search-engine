"""Perfis e ajustes finos compartilhados entre os motores de busca."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

SearchProfile = Literal["balanced", "lyrics", "metadata"]

_IDENTITY_MULTIPLIERS: dict[str, float] = {}

_TRACK_PROFILE_MULTIPLIERS: dict[SearchProfile, dict[str, float]] = {
    "balanced": _IDENTITY_MULTIPLIERS,
    "lyrics": {
        "track_name": 1.05,
        "artist_names": 0.8,
        "artist_genres": 0.75,
        "macro_genre": 0.75,
        "album_name": 0.7,
        "lyrics": 1.8,
    },
    "metadata": {
        "track_name": 1.45,
        "artist_names": 1.4,
        "artist_genres": 1.25,
        "macro_genre": 1.15,
        "album_name": 1.25,
        "lyrics": 0.55,
    },
}

_ENTITY_PROFILE_MULTIPLIERS: dict[str, dict[SearchProfile, dict[str, float]]] = {
    "artist": {
        "balanced": _IDENTITY_MULTIPLIERS,
        "lyrics": {
            "name": 0.55,
            "tagline": 1.6,
            "bio": 2.4,
            "raw_text": 4.0,
            "genres": 0.7,
            "origin": 0.6,
        },
        "metadata": {
            "name": 1.5,
            "tagline": 1.25,
            "bio": 0.8,
            "raw_text": 0.65,
            "genres": 1.35,
            "origin": 1.15,
        },
    },
    "album": {
        "balanced": _IDENTITY_MULTIPLIERS,
        "lyrics": {
            "title": 0.6,
            "artist": 0.75,
            "description": 3.0,
            "raw_text": 5.0,
        },
        "metadata": {
            "title": 1.45,
            "artist": 1.35,
            "description": 0.8,
            "raw_text": 0.65,
        },
    },
    "genre": {
        "balanced": _IDENTITY_MULTIPLIERS,
        "lyrics": {
            "name": 0.45,
            "description": 2.4,
            "raw_text": 5.0,
            "origin": 0.6,
            "representative_artists": 0.85,
        },
        "metadata": {
            "name": 1.45,
            "description": 0.8,
            "raw_text": 0.7,
            "origin": 1.2,
            "representative_artists": 1.35,
        },
    },
    "composer": {
        "balanced": _IDENTITY_MULTIPLIERS,
        "lyrics": {
            "name": 0.55,
            "bio": 2.2,
            "raw_text": 4.5,
            "genres": 0.75,
        },
        "metadata": {
            "name": 1.45,
            "bio": 0.85,
            "raw_text": 0.7,
            "genres": 1.2,
        },
    },
}


def _apply_profile_multipliers(
    base_weights: Mapping[str, float],
    multipliers: Mapping[str, float],
) -> dict[str, float]:
    return {
        field: float(weight) * float(multipliers.get(field, 1.0))
        for field, weight in base_weights.items()
        if float(weight) * float(multipliers.get(field, 1.0)) > 0
    }


def track_weights_for_profile(
    base_weights: Mapping[str, float],
    profile: SearchProfile = "balanced",
) -> dict[str, float]:
    return _apply_profile_multipliers(base_weights, _TRACK_PROFILE_MULTIPLIERS[profile])


def entity_weights_for_profile(
    kind: str,
    base_weights: Mapping[str, float],
    profile: SearchProfile = "balanced",
) -> dict[str, float]:
    multipliers = _ENTITY_PROFILE_MULTIPLIERS.get(kind, {}).get(profile, _IDENTITY_MULTIPLIERS)
    return _apply_profile_multipliers(base_weights, multipliers)
