from unittest.mock import patch

import httpx

from app.retrieval.embeddings import VOYAGE_EMBED_URL, VoyageEmbeddings


def test_voyage_retries_on_429():
    embedder = VoyageEmbeddings(
        api_key="test-key",
        model="voyage-3-lite",
        batch_size=8,
        batch_delay_s=0,
        max_retries=3,
        timeout_s=5,
    )
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited")
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2]}]},
        )

    with patch("app.retrieval.embeddings.time.sleep"):
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.side_effect = fake_post
            vector = embedder.embed_query("hello")
    assert vector == [0.1, 0.2]
    assert calls["n"] == 2
