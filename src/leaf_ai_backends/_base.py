from pydantic import BaseModel


class AIBackendConfig(BaseModel):
    pass


class AIBackend:
    config_cls = AIBackendConfig
    config: config_cls

    def __init__(self, config: config_cls):
        self.config = config


__all__ = ["AIBackendConfig", "AIBackend"]
