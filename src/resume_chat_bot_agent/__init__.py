def __getattr__(name):
    if name == "app":
        from .app import app as _app  # type: ignore
        return _app
    raise AttributeError(name)


