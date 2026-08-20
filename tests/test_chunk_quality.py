from app.retrieval.chunk_quality import is_low_content_chunk, quality_bonus


def test_toc_page_is_low_content():
    toc = "2 ALTA MERITA the OFFERING 6 the PROPERTY 42 competitive POSITIONING 64 the FINANCIALS 74"
    assert is_low_content_chunk(toc)
    assert quality_bonus(toc) < 0


def test_property_description_is_kept():
    prose = (
        "Alta Merita is a 2025-built, 260-unit apartment community located in Rancho Cucamonga, "
        "the Inland Empire's most affluent residential submarket. It was developed by Wood Partners."
    )
    assert not is_low_content_chunk(prose)
    assert quality_bonus(prose) > 0
