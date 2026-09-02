from __future__ import annotations

import json

import pytest

from spalmer.__main__ import _parameter_count, _prepare_data_parser, main


def test_parameter_count_accepts_experiment_scale_suffixes() -> None:
    assert _parameter_count("10m") == 10_000_000
    assert _parameter_count("1.5M") == 1_500_000
    assert _parameter_count("100,000") == 100_000


def test_plan_cli_requires_an_increasing_scale_dependent_vocabulary() -> None:
    with pytest.raises(SystemExit, match="vocabulary size must increase"):
        main(["plan", "10m", "50m", "--vocab-size", "8192", "8192"])


def test_plan_cli_emits_machine_readable_ladder(capsys) -> None:
    main(
        [
            "plan",
            "10m",
            "50m",
            "100m",
            "--vocab-size",
            "4096",
            "8192",
            "16384",
            "--expert-weight-format",
            "nvfp4",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert [item["target_parameters"] for item in payload] == [
        10_000_000,
        50_000_000,
        100_000_000,
    ]
    assert [item["config"]["vocab_size"] for item in payload] == [4096, 8192, 16384]
    assert {item["config"]["expert_weight_format"] for item in payload} == {"nvfp4"}


def test_prepare_data_cli_defaults_to_yylab_export_fields() -> None:
    args = _prepare_data_parser().parse_args(
        [
            "approved",
            "--output-directory",
            "prepared",
            "--name",
            "experiment",
            "--tokenizer-hf",
            "tokenizer",
        ]
    )

    assert args.kind_field == "domain"
    assert args.language_field == "lang"
    assert args.top_code_languages == 6
