"""Round-trip tests for the three SQLite-backed modules (temp databases)."""

import sqlite3

import pytest

import db
import generation_store
import moderation
import user_settings


def _use_temp_db(monkeypatch, tmp_path, module, filename):
    monkeypatch.setattr(module, "_db", db.Database(str(tmp_path / filename)))


def test_user_settings_round_trip(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path, user_settings, "settings.db")
    monkeypatch.setattr(user_settings, "_split_cache", set())

    user_settings.init_db()
    assert user_settings.get_settings(1)["sdxl_steps"] is None

    stored = user_settings.set_settings(1, sdxl_steps=30, negative_prompt="blurry")
    assert stored["sdxl_steps"] == 30
    assert user_settings.get_settings(1)["negative_prompt"] == "blurry"

    # Unknown keys are dropped instead of reaching the SQL statement.
    user_settings.set_settings(1, not_a_column="nope")
    assert user_settings.get_settings(1)["sdxl_steps"] == 30

    user_settings.mark_split_checkpoint("split.safetensors")
    assert user_settings.is_split_checkpoint("split.safetensors")

    user_settings.reset_settings(1)
    assert user_settings.get_settings(1)["sdxl_steps"] is None


def test_generation_store_round_trip(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path, generation_store, "generations.db")

    generation_store.init_db()
    generation_store.save(42, {"model": "sdxl", "seed": 7})
    assert generation_store.get(42) == {"model": "sdxl", "seed": 7}

    generation_store.pop(42)
    assert generation_store.get(42) is None


def test_generation_store_prunes_old_rows(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path, generation_store, "generations.db")

    generation_store.init_db()
    generation_store.save(1, {"model": "sdxl"})
    assert generation_store.prune(max_age_seconds=-1) == 1
    assert generation_store.get(1) is None


def test_moderation_admins_and_bans(monkeypatch, tmp_path):
    _use_temp_db(monkeypatch, tmp_path, moderation, "moderation.db")

    moderation.init_db()
    assert not moderation.is_admin(5)

    moderation.promote(5)
    assert moderation.is_admin(5) and moderation.get_admins() == [5]
    moderation.demote(5)
    assert not moderation.is_admin(5)

    moderation.ban(9)
    assert moderation.is_banned(9) and moderation.get_bans() == [9]
    moderation.unban(9)
    assert not moderation.is_banned(9)


def test_locked_commits_even_when_the_caller_forgets(tmp_path):
    path = str(tmp_path / "commit.db")
    database = db.Database(path)
    with database.locked() as conn:
        conn.execute("CREATE TABLE t (v TEXT)")

    # No explicit commit(): the context manager must persist the row.
    with database.locked() as conn:
        conn.execute("INSERT INTO t VALUES ('x')")

    with sqlite3.connect(path) as other:
        assert other.execute("SELECT v FROM t").fetchall() == [("x",)]


def test_locked_rolls_back_on_exception(tmp_path):
    path = str(tmp_path / "rollback.db")
    database = db.Database(path)
    with database.locked() as conn:
        conn.execute("CREATE TABLE t (v TEXT)")

    with pytest.raises(RuntimeError):
        with database.locked() as conn:
            conn.execute("INSERT INTO t VALUES ('y')")
            raise RuntimeError("boom")

    with sqlite3.connect(path) as other:
        assert other.execute("SELECT v FROM t").fetchall() == []
