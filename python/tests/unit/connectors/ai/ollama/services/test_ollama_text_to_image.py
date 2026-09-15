# Copyright (c) Microsoft. All rights reserved.

import base64
from unittest.mock import patch

import pytest
from ollama import GenerateResponse

from semantic_kernel.connectors.ai.ollama.ollama_prompt_execution_settings import (
    OllamaTextToImagePromptExecutionSettings,
)
from semantic_kernel.connectors.ai.ollama.services.ollama_text_to_image import OllamaTextToImage
from semantic_kernel.contents.image_content import ImageContent
from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError, ServiceInvalidResponseError

ENCODED_IMAGE = base64.b64encode(b"test_image_bytes").decode()


def sdk_response(image: str | None = ENCODED_IMAGE) -> GenerateResponse:
    """Build a response shaped like the one the Ollama SDK actually returns."""
    return GenerateResponse(model="test_model_id", created_at="2026-01-01T00:00:00Z", done=True, image=image)


def test_init_empty_service_id(model_id):
    """Test that the service initializes correctly with an empty service id."""
    ollama = OllamaTextToImage(ai_model_id=model_id)
    assert ollama.service_id == model_id


def test_custom_client(model_id, custom_client):
    """Test that the service initializes correctly with a custom client."""
    ollama = OllamaTextToImage(ai_model_id=model_id, client=custom_client)
    assert ollama.client == custom_client


def test_invalid_ollama_settings():
    """Test that the service initializes incorrectly with invalid settings."""
    with pytest.raises(ServiceInitializationError):
        _ = OllamaTextToImage(ai_model_id=123)


@pytest.mark.parametrize("exclude_list", [["OLLAMA_IMAGE_MODEL_ID"]], indirect=True)
def test_init_empty_model_id(ollama_unit_test_env):
    """Test that the service initializes incorrectly with an empty model id."""
    with pytest.raises(ServiceInitializationError):
        _ = OllamaTextToImage(env_file_path="fake_env_file_path.env")


@patch("ollama.AsyncClient.__init__", return_value=None)  # mock_client
@patch("ollama.AsyncClient.generate")  # mock_generate
async def test_custom_host(mock_generate, mock_client, model_id, host, prompt):
    """Test that the service generates an image correctly with a custom host."""
    mock_generate.return_value = {"image": ENCODED_IMAGE}

    ollama = OllamaTextToImage(ai_model_id=model_id, host=host)
    _ = await ollama.generate_image(prompt)

    mock_client.assert_called_once_with(host=host)


@patch("ollama.AsyncClient.generate")
async def test_generate_image(mock_generate, model_id, prompt):
    """Test that the service decodes the base64 image returned by Ollama."""
    mock_generate.return_value = {"image": ENCODED_IMAGE}
    settings = OllamaTextToImagePromptExecutionSettings()
    settings.options = {"test_key": "test_value"}

    ollama = OllamaTextToImage(ai_model_id=model_id)
    image = await ollama.generate_image(prompt, settings=settings)

    assert image == b"test_image_bytes"
    mock_generate.assert_called_once_with(
        model=model_id,
        prompt=prompt,
        stream=False,
        options={"test_key": "test_value"},
    )


@patch("ollama.AsyncClient.generate")
async def test_get_image_content(mock_generate, model_id, prompt):
    """Test that the inherited get_image_content returns ImageContent with the image data."""
    mock_generate.return_value = {"image": ENCODED_IMAGE}

    ollama = OllamaTextToImage(ai_model_id=model_id)
    content = await ollama.get_image_content(prompt, OllamaTextToImagePromptExecutionSettings())

    assert isinstance(content, ImageContent)
    assert content.data == b"test_image_bytes"


@patch("ollama.AsyncClient.generate")
async def test_generate_image_with_size_settings(mock_generate, model_id, prompt):
    """Test that width, height and steps from the settings are forwarded to Ollama."""
    mock_generate.return_value = {"image": ENCODED_IMAGE}
    settings = OllamaTextToImagePromptExecutionSettings(width=512, height=256, steps=4)

    ollama = OllamaTextToImage(ai_model_id=model_id)
    _ = await ollama.generate_image(prompt, settings=settings)

    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["width"] == 512
    assert call_kwargs["height"] == 256
    assert call_kwargs["steps"] == 4


@patch("ollama.AsyncClient.generate")
async def test_generate_image_deprecated_width_and_height_arguments(mock_generate, model_id, prompt):
    """Test that the deprecated width and height arguments still reach Ollama, with a warning."""
    mock_generate.return_value = {"image": ENCODED_IMAGE}

    ollama = OllamaTextToImage(ai_model_id=model_id)
    with pytest.warns(DeprecationWarning):
        image = await ollama.generate_image(prompt, width=512, height=256)

    assert image == b"test_image_bytes"
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["width"] == 512
    assert call_kwargs["height"] == 256


@patch("ollama.AsyncClient.generate")
async def test_generate_image_settings_take_precedence_over_arguments(mock_generate, model_id, prompt):
    """Test that explicit settings win over the deprecated width and height arguments."""
    mock_generate.return_value = {"image": ENCODED_IMAGE}
    settings = OllamaTextToImagePromptExecutionSettings(width=1024, height=1024)

    ollama = OllamaTextToImage(ai_model_id=model_id)
    with pytest.warns(DeprecationWarning):
        _ = await ollama.generate_image(prompt, width=512, height=256, settings=settings)

    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["width"] == 1024
    assert call_kwargs["height"] == 1024


@patch("ollama.AsyncClient.generate")
async def test_generate_image_without_image_in_response(mock_generate, model_id, prompt):
    """Test that a response without image data raises instead of returning empty bytes."""
    mock_generate.return_value = {"response": "this model returns text"}

    ollama = OllamaTextToImage(ai_model_id=model_id)
    with pytest.raises(ServiceInvalidResponseError):
        await ollama.generate_image(prompt)


@patch("ollama.AsyncClient.generate")
async def test_generate_image_sdk_response(mock_generate, model_id, prompt):
    """Test decoding against a GenerateResponse, which is what the SDK returns."""
    mock_generate.return_value = sdk_response()

    ollama = OllamaTextToImage(ai_model_id=model_id)
    image = await ollama.generate_image(prompt)

    assert image == b"test_image_bytes"


@patch("ollama.AsyncClient.generate")
async def test_generate_image_sdk_response_without_image(mock_generate, model_id, prompt):
    """Test that a GenerateResponse carrying no image raises."""
    mock_generate.return_value = sdk_response(image=None)

    ollama = OllamaTextToImage(ai_model_id=model_id)
    with pytest.raises(ServiceInvalidResponseError):
        await ollama.generate_image(prompt)


@patch("ollama.AsyncClient.generate")
async def test_generate_image_typed_settings_are_not_repacked(mock_generate, model_id, prompt):
    """Test that a cleared field on a reused settings object is not restored from extension data."""
    mock_generate.return_value = sdk_response()
    settings = OllamaTextToImagePromptExecutionSettings(width=512)

    ollama = OllamaTextToImage(ai_model_id=model_id)
    await ollama.generate_image(prompt, settings=settings)
    assert mock_generate.call_args.kwargs["width"] == 512

    settings.width = None
    await ollama.generate_image(prompt, settings=settings)
    assert "width" not in mock_generate.call_args.kwargs


@patch("ollama.AsyncClient.generate")
async def test_generate_image_kwargs_do_not_collide_with_request_keys(mock_generate, model_id, prompt):
    """Test that request-control keys passed as kwargs do not raise TypeError."""
    mock_generate.return_value = sdk_response()

    ollama = OllamaTextToImage(ai_model_id=model_id)
    image = await ollama.generate_image(prompt, stream=True, model="other_model")

    assert image == b"test_image_bytes"
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["stream"] is False
    assert call_kwargs["model"] == model_id


@patch("ollama.AsyncClient.generate")
async def test_generate_image_settings_are_not_mutated(mock_generate, model_id, prompt):
    """Test that the deprecated size arguments do not mutate the caller's settings object."""
    mock_generate.return_value = sdk_response()
    settings = OllamaTextToImagePromptExecutionSettings()

    ollama = OllamaTextToImage(ai_model_id=model_id)
    with pytest.warns(DeprecationWarning):
        await ollama.generate_image(prompt, width=512, height=256, settings=settings)

    assert settings.width is None
    assert settings.height is None
    assert mock_generate.call_args.kwargs["width"] == 512
