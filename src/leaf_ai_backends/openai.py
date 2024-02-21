import os
from abc import abstractmethod
from enum import Enum
from typing import Any, Literal, Optional, Union

import openai
from openai import DEFAULT_MAX_RETRIES
from openai._types import NOT_GIVEN
from pydantic import BaseModel, Field, PrivateAttr

from ._base import AIBackend, AIBackendConfig


LOOK_ENVIRONMENT_VARIABLES = "look_environment_variables"


class OpenAIEnvironmentVariable(Enum):
    API_KEY = "OPENAI_API_KEY"
    AZURE_API_KEY = "AZURE_OPENAI_API_KEY"
    ORGANIZATION = "OPENAI_ORG_ID"
    AZURE_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
    API_VERSION = "OPENAI_API_VERSION"


class _BaseOpenAIClientConfig(BaseModel):
    api_key: Optional[str] = Field(default=None)
    organization: Optional[str] = Field(default=None)
    chat_model: Optional[str] = Field(default=None)
    completion_model: Optional[str] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)
    vision_model: Optional[str] = Field(default=None)
    image_generation_model: Optional[str] = Field(default=None)
    tts_model: Optional[str] = Field(default=None)
    whisper_model: Optional[str] = Field(default=None)
    moderation_model: Optional[str] = Field(default=None)

    _is_azure: bool = PrivateAttr(default=False)

    @property
    def is_azure(self) -> bool:
        return self._is_azure

    def model_post_init(self, __context: Any) -> None:
        if not self.api_key:
            env = OpenAIEnvironmentVariable.AZURE_API_KEY if self.is_azure else OpenAIEnvironmentVariable.API_KEY
            api_key = os.environ.get(env.value, None)
            if not api_key:
                raise ValueError(
                    f"must specify 'api_key' argument or "
                    f"'{OpenAIEnvironmentVariable.API_KEY.value}' environment variable"
                )
            self.api_key = api_key
        if not self.organization:
            self.organization = os.environ.get(OpenAIEnvironmentVariable.ORGANIZATION.value, None)

    @abstractmethod
    def prepare_client_init_kwargs(self, **kwargs) -> dict:
        """Prepare the kwargs for initializing the client

        :param kwargs: Additional kwargs that not be contained in the client data for initializing the client
        :return: A dict of kwargs for initializing the client
        :rtype: dict
        """
        init_kwargs = {"api_key": self.api_key, "organization": self.organization}
        init_kwargs.update(timeout=kwargs.get("timeout", NOT_GIVEN))
        init_kwargs.update(max_retries=kwargs.get("max_retries", DEFAULT_MAX_RETRIES))
        init_kwargs.update(default_headers=kwargs.get("default_headers", None))
        init_kwargs.update(default_query=kwargs.get("default_query", None))
        init_kwargs.update(http_client=kwargs.get("http_client", None))
        init_kwargs.update(_strict_response_validation=kwargs.get("_strict_response_validation", False))
        return init_kwargs


class OpenAIClientConfig(_BaseOpenAIClientConfig):
    base_url: Optional[str] = Field(default=None)
    chat_model: Optional[
        Literal[
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        ]
    ] = Field(default=None)
    completion_model: Literal["gpt-3.5-turbo-instruct"] = Field(default="gpt-3.5-turbo-instruct")
    embedding_model: Optional[
        Literal[
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ]
    ] = Field(default=None)
    vision_model: Optional[Literal["gpt-4-1106-vision-preview"]] = Field(default=None)
    image_generation_model: Optional[
        Literal[
            "dall-e-2",
            "dall-e-3",
        ]
    ] = Field(default=None)
    tts_model: Optional[
        Literal[
            "tts-1",
            "tts-1-hd",
        ]
    ] = Field(default=None)
    whisper_model: Literal["whisper-1"] = Field(default="whisper-1")
    moderation_model: Literal[
        "text-moderation-latest",
        "text-moderation-stable",
    ] = Field(default="text-moderation-latest")

    def prepare_client_init_kwargs(self, **kwargs) -> dict:
        init_kwargs = super().prepare_client_init_kwargs(**kwargs)
        init_kwargs.update(base_url=self.base_url)
        return init_kwargs


class AzureOpenAIClientConfig(_BaseOpenAIClientConfig):
    # NOTE: we currently not support using 'azure_ad_token' and 'azure_ad_token_provider'
    base_url: Optional[str] = Field(default=None)
    azure_endpoint: Optional[str] = Field(default=None)
    azure_deployment: Optional[str] = Field(default=None)
    api_version: str = Field(default=LOOK_ENVIRONMENT_VARIABLES)

    _is_azure: bool = PrivateAttr(default=True)

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if not self.azure_endpoint:
            self.azure_endpoint = os.environ.get(OpenAIEnvironmentVariable.AZURE_ENDPOINT.value, None)
        if not self.base_url and not self.azure_endpoint:
            raise ValueError(
                "Must provide one of the `base_url` or `azure_endpoint` arguments, "
                "or the `AZURE_OPENAI_ENDPOINT` environment variable"
            )
        if not self.api_version:
            self.api_version = os.environ.get(OpenAIEnvironmentVariable.API_VERSION.value, None)
            if not self.api_version:
                raise ValueError(
                    "Must provide the `api_version` argument, or the `OPENAI_API_VERSION` environment variable"
                )

    def prepare_client_init_kwargs(self, **kwargs) -> dict:
        init_kwargs = super().prepare_client_init_kwargs(**kwargs)
        if self.base_url:
            init_kwargs.update(base_url=self.base_url)
        else:
            init_kwargs.update(azure_endpoint=self.azure_endpoint)
            if self.azure_deployment:
                init_kwargs.update(azure_deployment=self.azure_deployment)
        init_kwargs.update(api_version=self.api_version)

        return init_kwargs


class OpenAIBackendConfig(AIBackendConfig):
    client_config: Union[OpenAIClientConfig, AzureOpenAIClientConfig] = Field(default=..., union_mode="smart")

    def _prepare_oai_client(self, is_async: bool) -> Union[
        openai.OpenAI, openai.AsyncOpenAI, openai.AzureOpenAI, openai.AsyncAzureOpenAI
    ]:
        return (
            (openai.AsyncAzureOpenAI if is_async else openai.AzureOpenAI)
            if self.is_azure_openai else
            (openai.AsyncOpenAI if is_async else openai.OpenAI)
        )(**self.client_init_kwargs)

    def get_client(self) -> Union[openai.OpenAI, openai.AzureOpenAI]:
        return self._prepare_oai_client(False)

    def get_async_client(self) -> Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]:
        return self._prepare_oai_client(True)

    @property
    def is_azure_openai(self) -> bool:
        return self.client_config.is_azure

    @property
    def client_init_kwargs(self) -> dict:
        return self.client_config.prepare_client_init_kwargs()

    @property
    def chat_model(self) -> Optional[str]:
        return self.client_config.chat_model

    @property
    def completion_model(self) -> Optional[str]:
        return self.client_config.completion_model

    @property
    def embedding_model(self) -> Optional[str]:
        return self.client_config.embedding_model

    @property
    def image_generation_model(self) -> Optional[str]:
        return self.client_config.image_generation_model

    @property
    def vision_model(self) -> Optional[str]:
        return self.client_config.vision_model

    @property
    def tts_model(self) -> Optional[str]:
        return self.client_config.tts_model

    @property
    def whisper_model(self) -> Optional[str]:
        return self.client_config.whisper_model

    @property
    def moderation_model(self) -> Optional[str]:
        return self.client_config.moderation_model


class OpenAIBackend(AIBackend):
    config_cls = OpenAIBackendConfig
    config: config_cls

    def __init__(self, config: config_cls):
        super().__init__(config)

    @property
    def client(self) -> Union[openai.OpenAI, openai.AzureOpenAI]:
        return self.config.get_client()

    @property
    def async_client(self) -> Union[openai.AsyncOpenAI, openai.AsyncAzureOpenAI]:
        return self.config.get_async_client()


__all__ = ["OpenAIClientConfig", "AzureOpenAIClientConfig", "OpenAIBackendConfig", "OpenAIBackend"]
