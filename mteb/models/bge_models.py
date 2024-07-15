from __future__ import annotations

import logging
from functools import partial
from typing import Any

import numpy as np

from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts
from mteb.requires_package import requires_package

logger = logging.getLogger(__name__)

class BGEWrapper:
    def __init__(self, model_name: str, embed_dim: int | None = None, **kwargs) -> None:
        from google.cloud import aiplatform
        self._client = aiplatform.Endpoint("366038416002908160")
        self._model_name = model_name
        self._embed_dim = embed_dim

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 250,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        embeddings = []
        for batch in range(0, len(sentences), batch_size):
            text_batch = sentences[batch : batch + batch_size]
            instances = [[text] for text in text_batch]
            prediction = self._client.predict(instances=instances)
            embeddings.extend(prediction.predictions[0][0])

        return np.array(embeddings)

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 250,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.encode(
            queries, batch_size=batch_size, prompt_name=prompt_name, **kwargs,
        )

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list[str]],
        batch_size: int = 250,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus)
        return self.encode(
            sentences, batch_size=batch_size, prompt_name=prompt_name, **kwargs,
        )

bge_m3 = ModelMeta(
    loader=partial(BGEWrapper, model_name="BAAI/bge-m3"),  # type: ignore
    name="BAAI/bge-m3",
    languages=["eng_Latn"],
    open_source=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
)

bge_base_en_v1_5 = ModelMeta(
    loader=partial(BGEWrapper, model_name="BAAI/bge-base-en-v1.5"),  # type: ignore
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(BGEWrapper, model_name="BAAI/bge-large-en-v1.5"),  # type: ignore
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_source=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
)
