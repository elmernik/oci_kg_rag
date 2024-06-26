from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator


VALID_PROVIDERS = ("cohere", "meta")

class OCIAuthType(Enum):
    """OCI authentication types as enumerator."""

    API_KEY = 1
    SECURITY_TOKEN = 2
    INSTANCE_PRINCIPAL = 3
    RESOURCE_PRINCIPAL = 4


class OCIGenAIChatBase(BaseModel, ABC):
    """Base class for new OCI GenAI chat models"""

    client: Any  #: :meta private:

    auth_type: Optional[str] = "API_KEY"
    """Authentication type. Only supports API_KEY currently, might add the rest later.
    """

    auth_profile: Optional[str] = "DEFAULT"
    """The name of the profile in ~/.oci/config
    If not specified , DEFAULT will be used 
    """

    model_id: str = None  # type: ignore[assignment]
    """Id of the model to call, e.g., ocid1.generativeaimodel...something"""

    provider: str = None  # type: ignore[assignment]
    """Provider name of the model. Requires user input
    """

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model"""

    service_endpoint: str = None  # type: ignore[assignment]
    """service endpoint url"""

    compartment_id: str = None  # type: ignore[assignment]
    """OCID of compartment"""

    is_stream: bool = False
    """Whether to stream back partial progress"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that OCI config and python package exists in environment."""

        # Skip creating new client if passed in constructor
        if values["client"] is not None:
            return values

        try:
            import oci

            client_kwargs = {
                "config": {},
                "signer": None,
                "service_endpoint": values["service_endpoint"],
                "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
                "timeout": (10, 240),  # default timeout config for OCI Gen AI service
            }

            if values["auth_type"] == OCIAuthType(1).name:
                client_kwargs["config"] = oci.config.from_file(
                    profile_name=values["auth_profile"]
                )
                client_kwargs.pop("signer", None)
            else:
                raise ValueError("Please provide valid value to auth_type")

            values["client"] = oci.generative_ai_inference.GenerativeAiInferenceClient(
                **client_kwargs
            )

        except ImportError as ex:
            raise ImportError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex
        except Exception as e:
            raise ValueError(
                "Could not authenticate with OCI client. "
                "Please check if ~/.oci/config exists. "
                "If INSTANCE_PRINCIPLE or RESOURCE_PRINCIPLE is used, "
                "Please check the specified "
                "auth_profile and auth_type are valid."
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _get_provider(self) -> str:
        if self.provider is not None:
            provider = self.provider
        else:
            provider = self.model_id.split(".")[0].lower()

        if provider not in VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider derived from model_id: {self.model_id} "
                "Please explicitly pass in the supported provider "
                "when using custom endpoint"
            )
        return provider


class OCIGenAIChat(LLM, OCIGenAIChatBase):
    """OCI large language models.

    To authenticate, the OCI client uses the methods described in
    https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdk_authentication_methods.htm

    The authentifcation method is passed through auth_type and should be one of:
    API_KEY (default), Might add the rest later

    Make sure you have the required policies (profile/roles) to
    access the OCI Generative AI service.
    If a specific config profile is used, you must pass
    the name of the profile (from ~/.oci/config) through auth_profile.

    To use, you must provide the compartment id
    along with the endpoint url, and model id
    as named parameters to the constructor.

    Example:
        .. code-block:: python

            from oci_generative_ai_chat import OCIGenAIChat

            llm = OCIGenAIChat(
                provider="cohere",
                model_id="MY_MODEL_ID",
                service_endpoint="https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
                compartment_id="MY_COMPARTMENT_ID",
                model_kwargs={"temperature": 0, "max_tokens": 500},
            )

    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ocichat"
    

    def _prepare_chat_request(
            self, prompt: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from oci.generative_ai_inference import models
        """Prepare Chat Request depending on whether using Cohere or Llama, 
        check out Oracle docs for API reference"""
        _model_kwargs = self.model_kwargs or {}

        chat_params = {**_model_kwargs, **kwargs}
        chat_params["is_stream"] = self.is_stream

        provider = self.provider
        if provider == "cohere":
            chat_params["message"] = prompt
            chat_request = models.CohereChatRequest(**chat_params)
        elif provider == "meta":
            content = models.TextContent()
            content.text = prompt
            message = models.Message()
            message.role = "USER"
            message.content = [content]
            chat_params["api_format"] = models.BaseChatRequest.API_FORMAT_GENERIC
            chat_params["messages"] = [message]
            chat_request = models.GenericChatRequest(**chat_params)
        else:
            raise ValueError(f"Invalid provider: {provider}")

        return chat_request
    

    def _process_response(self, response: Any) -> str:
        """Return text content from LLM response"""
        provider = self.provider
        if provider == "cohere":
            return vars(response)["data"].chat_response.text
        elif provider == "meta":
            return vars(response)["data"].chat_response.choices[0].message.content[0].text # This is a nightmare
        else:
            raise ValueError(f"Invalid provider: {provider}")


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from oci.generative_ai_inference import models
        """Call out to OCIGenAIChat generate endpoint.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = llm.invoke("Tell me a joke.")
        """
        
        # Check oci python package documentation for more information about this
        chat_request = self._prepare_chat_request(prompt, kwargs)

        serving_mode = models.OnDemandServingMode(model_id=self.model_id)
        chat_detail = models.ChatDetails(serving_mode=serving_mode, chat_request=chat_request, compartment_id=self.compartment_id)

        response = self.client.chat(chat_detail)
        return self._process_response(response)