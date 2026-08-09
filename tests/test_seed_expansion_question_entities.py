"""Tests for method-like names extracted from retrieval questions."""

from __future__ import annotations

import pytest

from littraceqa.di_pipeline.retrieve.seed_expansion.question_entities import (
    is_generic_alias,
    question_aliases,
)


@pytest.mark.parametrize("alias", ["ACL", "acl", "LoRA", "lora", "RAG", "rag"])
def test_generic_alias_matching_is_case_insensitive(alias):
    assert is_generic_alias(alias)


@pytest.mark.parametrize("alias", ["TCM", "D-FINE", "sCT"])
def test_distinctive_method_aliases_are_not_generic(alias):
    assert not is_generic_alias(alias)


def test_question_aliases_exclude_generic_names_but_keep_distinctive_methods():
    question = "Compare ACL papers about LoRA and RAG with TCM and D-FINE."

    assert question_aliases(question) == ("TCM", "D-FINE")
