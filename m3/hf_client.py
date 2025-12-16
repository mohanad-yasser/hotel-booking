# hf_client.py

import os
from huggingface_hub import InferenceClient
from config import HF_API_TOKEN, HF_MODEL_ID


if HF_API_TOKEN is None:
    raise ValueError(
        "HF_API_TOKEN is not set. "
        "Add HF_API_TOKEN=... to your config.txt or environment."
    )

if HF_MODEL_ID is None:
    raise ValueError(
        "HF_MODEL_ID is not set. "
        "Add HF_MODEL_ID=... to your config.txt."
    )


# Create a global inference client
client = InferenceClient(api_key=HF_API_TOKEN)


def call_hf_llm(prompt: str) -> str:
    """
    Call a Hugging Face chat/instruction model using the new Inference API.
    This uses the 'chat_completion' endpoint because many free models,
    including Llama 3, are exposed as conversational/chat models.
    """

    # Chat style call
    completion = client.chat_completion(
        model=HF_MODEL_ID,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
    )

    # completion.choices is a list, we take the first answer
    content = completion.choices[0].message["content"]
    return content