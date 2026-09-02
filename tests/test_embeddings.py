from sentinelrag.embeddings import cosine, local_embed


def test_local_embeddings_are_normalized_and_relevant() -> None:
    remote = local_embed("employees may work remotely")
    same_topic = local_embed("remote work for employees")
    unrelated = local_embed("hotel receipts and domestic flights")
    assert round(cosine(remote, remote), 5) == 1.0
    assert cosine(remote, same_topic) > cosine(remote, unrelated)
