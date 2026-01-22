import pytest
from probity.datasets.templated import (
    TemplateVariable,
    Template,
    TemplatedDataset,
)
from probity.datasets.base import ProbingDataset, ProbingExample
from probity.datasets.position_finder import Position


# Sample data for testing
adj_var = TemplateVariable(
    name="ADJ",
    values=["good", "bad", "great", "terrible"],
    attributes={"sentiment": ["positive", "negative", "positive", "negative"]},
    class_bound=True,
    class_key="sentiment",
)
verb_var = TemplateVariable(
    name="VERB",
    values=["loved", "hated", "enjoyed", "disliked"],
    attributes={"sentiment": ["positive", "negative", "positive", "negative"]},
    class_bound=True,
    class_key="sentiment",
)
noun_var = TemplateVariable(name="NOUN", values=["movie", "film"], attributes=None)

template1 = Template(
    template="I thought this {NOUN} was {ADJ}, I {VERB} it.",
    variables={"ADJ": adj_var, "VERB": verb_var, "NOUN": noun_var},
    attributes={"task": "sentiment"},
)

template_no_bound = Template(
    template="{ADJ} {NOUN}",
    variables={"ADJ": adj_var, "NOUN": noun_var},
)


# --- Test TemplateVariable ---
def test_template_variable_init():
    var = TemplateVariable(name="TEST", values=["a", "b"])
    assert var.name == "TEST"
    assert var.values == ["a", "b"]
    assert var.attributes is None
    assert not var.class_bound
    assert var.class_key is None


def test_template_variable_init_with_attrs():
    var = TemplateVariable(
        name="TEST",
        values=["a", "b"],
        attributes={"type": ["X", "Y"]},
        class_bound=True,
        class_key="type",
    )
    assert var.name == "TEST"
    assert var.attributes == {"type": ["X", "Y"]}
    assert var.class_bound
    assert var.class_key == "type"


# --- Test Template ---
def test_template_init():
    t = Template(template="{VAR}", variables={"VAR": adj_var})
    assert t.template == "{VAR}"
    assert t.variables == {"VAR": adj_var}
    assert t.attributes is None


def test_template_get_marker():
    assert template1.get_marker("ADJ") == "{ADJ}"


def test_template_get_all_markers():
    assert template1.get_all_markers() == {
        "ADJ": "{ADJ}",
        "VERB": "{VERB}",
        "NOUN": "{NOUN}",
    }


def test_template_validate_valid():
    assert template1.validate()


def test_template_validate_invalid_missing_var():
    invalid_template = Template(template="{ADJ} {MISSING}", variables={"ADJ": adj_var})
    assert not invalid_template.validate()


def test_template_validate_invalid_extra_var():
    invalid_template = Template(
        template="{ADJ}", variables={"ADJ": adj_var, "EXTRA": noun_var}
    )
    assert not invalid_template.validate()


# --- Test TemplatedDataset ---
def test_templated_dataset_init_valid():
    ds = TemplatedDataset(templates=[template1])
    assert ds.templates == [template1]
    assert ds.attributes == {}


def test_templated_dataset_init_invalid_template():
    invalid_template = Template(template="{ADJ} {MISSING}", variables={"ADJ": adj_var})
    with pytest.raises(ValueError):
        TemplatedDataset(templates=[invalid_template])


def test_to_probing_dataset_basic():
    ds = TemplatedDataset(templates=[template1])
    probing_ds = ds.to_probing_dataset(auto_add_positions=False)
    assert isinstance(probing_ds, ProbingDataset)
    # Expected examples: 2 (NOUN) * (2 pos * 2 pos + 2 neg * 2 neg) = 2 * (4 + 4) = 16
    assert len(probing_ds.examples) == 16


def test_to_probing_dataset_with_labels():
    ds = TemplatedDataset(templates=[template1])
    probing_ds = ds.to_probing_dataset(
        label_from_attributes="sentiment",
        label_map={"positive": 1, "negative": 0},
        auto_add_positions=False,
    )
    assert len(probing_ds.examples) == 16
    positive_count = sum(1 for ex in probing_ds.examples if ex.label == 1)
    negative_count = sum(1 for ex in probing_ds.examples if ex.label == 0)
    # Each noun (movie, film) combines with (pos_adj, pos_verb) and (neg_adj, neg_verb)
    # pos combinations = 2 adj * 2 verb = 4 per noun -> 8 total positive
    # neg combinations = 2 adj * 2 verb = 4 per noun -> 8 total negative
    assert positive_count == 8
    assert negative_count == 8
    assert probing_ds.examples[0].label_text in ["positive", "negative"]


def test_to_probing_dataset_attribute_slicing():
    """Verify that variable attributes are correctly sliced per example."""
    ds = TemplatedDataset(templates=[template1])
    probing_ds = ds.to_probing_dataset(
        label_from_attributes="sentiment",
        label_map={"positive": 1, "negative": 0},
        auto_add_positions=False,
    )
    # Check an example
    example = next(
        ex for ex in probing_ds.examples if "I thought this movie was good" in ex.text
    )  # Should be positive
    assert example is not None
    assert example.label == 1
    assert example.attributes["class"] == "positive"
    assert example.attributes["variables"]["ADJ"] == {"sentiment": "positive"}
    assert example.attributes["variables"]["VERB"] == {"sentiment": "positive"}
    assert (
        example.attributes["variables"]["NOUN"] is None
    )  # No attributes defined for NOUN


def test_to_probing_dataset_auto_positions():
    ds = TemplatedDataset(templates=[template1])
    probing_ds = ds.to_probing_dataset(auto_add_positions=True)
    assert len(probing_ds.examples) == 16
    assert probing_ds.position_types == {"ADJ", "VERB", "NOUN"}
    example = probing_ds.examples[0]
    # Explicitly check keys to avoid potential issues with `in` on custom class
    assert example.character_positions is not None
    assert "ADJ" in example.character_positions.keys()
    assert "VERB" in example.character_positions.keys()
    assert "NOUN" in example.character_positions.keys()
    # Check position calculation for a specific example
    # Example: "I thought this movie was good, I loved it."
    example = next(
        ex
        for ex in probing_ds.examples
        if ex.text == "I thought this movie was good, I loved it."
    )
    adj_pos = example.character_positions["ADJ"]
    verb_pos = example.character_positions["VERB"]
    noun_pos = example.character_positions["NOUN"]
    assert isinstance(adj_pos, Position)
    assert isinstance(verb_pos, Position)
    assert isinstance(noun_pos, Position)
    assert example.text[adj_pos.start : adj_pos.end] == "good"
    assert example.text[verb_pos.start : verb_pos.end] == "loved"
    assert example.text[noun_pos.start : noun_pos.end] == "movie"


def test_to_probing_dataset_no_bound_vars():
    # Create a template without class-bound variables
    adj_no_bound = TemplateVariable(name="ADJ", values=["hot", "cold"])
    noun_no_bound = TemplateVariable(name="NOUN", values=["day", "night"])
    template_nb = Template(
        template="It is a {ADJ} {NOUN}.",
        variables={"ADJ": adj_no_bound, "NOUN": noun_no_bound},
    )
    ds = TemplatedDataset(templates=[template_nb])
    probing_ds = ds.to_probing_dataset(auto_add_positions=False)
    # Expected: 2 adj * 2 noun = 4 examples
    assert len(probing_ds.examples) == 4
    texts = {ex.text for ex in probing_ds.examples}
    assert texts == {
        "It is a hot day.",
        "It is a hot night.",
        "It is a cold day.",
        "It is a cold night.",
    }
    # No class should be assigned if no class_bound vars
    assert all(ex.attributes["class"] is None for ex in probing_ds.examples)


def test_from_movie_sentiment_template():
    adjectives = {"positive": ["great", "excellent"], "negative": ["bad", "awful"]}
    verbs = {"positive": ["loved", "adored"], "negative": ["hated", "despised"]}
    ds = TemplatedDataset.from_movie_sentiment_template(adjectives, verbs)
    assert len(ds.templates) == 1
    template = ds.templates[0]
    assert template.template == "I thought this movie was {ADJ}, I {VERB} it."
    assert "ADJ" in template.variables
    assert "VERB" in template.variables
    assert template.variables["ADJ"].class_bound
    assert template.variables["VERB"].class_bound
    probing_ds = ds.to_probing_dataset(
        label_from_attributes="sentiment",
        label_map={"positive": 1, "negative": 0},
        auto_add_positions=False,
    )
    # Expected: 2 pos_adj * 2 pos_verb + 2 neg_adj * 2 neg_verb = 4 + 4 = 8
    assert len(probing_ds.examples) == 8
    positive_count = sum(1 for ex in probing_ds.examples if ex.label == 1)
    negative_count = sum(1 for ex in probing_ds.examples if ex.label == 0)
    assert positive_count == 4
    assert negative_count == 4


def test_from_mood_story_template():
    names = ["Alice", "Bob"]
    verbs = {"positive": ["loves", "enjoys"], "negative": ["hates", "dislikes"]}
    ds = TemplatedDataset.from_mood_story_template(names, verbs)
    assert len(ds.templates) == 1
    template = ds.templates[0]
    assert template.template == "{NAME} {VERB} parties, and does so whenever possible."
    assert "NAME" in template.variables
    assert "VERB" in template.variables
    assert not template.variables["NAME"].class_bound
    assert not template.variables[
        "VERB"
    ].class_bound  # NOTE: In this factory, VERB isn't class-bound in the *template* definition itself
    probing_ds = ds.to_probing_dataset(auto_add_positions=False)
    # Expected: 2 names * 4 verbs = 8 examples
    assert len(probing_ds.examples) == 8
    texts = {ex.text for ex in probing_ds.examples}
    expected_texts = {
        f"{name} {verb} parties, and does so whenever possible."
        for name in names
        for verb_list in verbs.values()
        for verb in verb_list
    }
    assert texts == expected_texts
    # Verify verb attributes are stored correctly in the example
    example_alice_loves = next(
        ex for ex in probing_ds.examples if "Alice loves" in ex.text
    )
    assert (
        example_alice_loves.attributes["variables"]["VERB"]["sentiment"] == "positive"
    )
    example_bob_hates = next(ex for ex in probing_ds.examples if "Bob hates" in ex.text)
    assert example_bob_hates.attributes["variables"]["VERB"]["sentiment"] == "negative"
