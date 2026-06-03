"""Smoke tests: the package and its core modules import cleanly."""


def test_package_imports():
    import dataforager  # noqa: F401


def test_hyse_module_imports():
    from dataforager.hyse import hypo_schema_search  # noqa: F401


def test_api_app_imports():
    from dataforager.api import app  # noqa: F401

    assert hasattr(app, "app")  # the Flask application object
    assert callable(app.main)  # console-script entry point
