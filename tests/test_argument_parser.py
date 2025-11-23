"""Tests for the fallback HfArgumentParser when transformers is not available."""
import sys
from dataclasses import dataclass, field
from typing import Optional

# Mock the transformers import to force using fallback
sys.modules['transformers'] = None

# Now import after mocking
from syntk.argument_parser import HfArgumentParser, _get_base_type  # noqa: E402


class TestGetBaseType:
    """Tests for _get_base_type helper function."""

    def test_extracts_type_from_optional_str(self):
        """Test extracting str from Optional[str]."""
        result = _get_base_type(Optional[str])
        assert result is str

    def test_extracts_type_from_optional_int(self):
        """Test extracting int from Optional[int]."""
        result = _get_base_type(Optional[int])
        assert result is int

    def test_extracts_type_from_optional_float(self):
        """Test extracting float from Optional[float]."""
        result = _get_base_type(Optional[float])
        assert result is float

    def test_returns_direct_type_str(self):
        """Test that direct str type is returned as-is."""
        result = _get_base_type(str)
        assert result is str

    def test_returns_direct_type_int(self):
        """Test that direct int type is returned as-is."""
        result = _get_base_type(int)
        assert result is int

    def test_returns_direct_type_float(self):
        """Test that direct float type is returned as-is."""
        result = _get_base_type(float)
        assert result is float


class TestHfArgumentParserFallback:
    """Tests for the fallback HfArgumentParser implementation."""

    def test_parses_string_field(self):
        """Test parsing a simple string field."""
        @dataclass
        class Args:
            name: str = field(default="default_name", metadata={"help": "Name"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_args_into_dataclasses(args=["--name", "test"])

        assert args.name == "test"

    def test_parses_int_field(self):
        """Test parsing an integer field."""
        @dataclass
        class Args:
            count: int = field(default=10, metadata={"help": "Count"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_args_into_dataclasses(args=["--count", "42"])

        assert args.count == 42
        assert isinstance(args.count, int)

    def test_parses_float_field(self):
        """Test parsing a float field."""
        @dataclass
        class Args:
            value: float = field(default=1.0, metadata={"help": "Value"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_args_into_dataclasses(args=["--value", "3.14"])

        assert args.value == 3.14
        assert isinstance(args.value, float)

    def test_parses_optional_string_field(self):
        """Test parsing an Optional[str] field."""
        @dataclass
        class Args:
            name: Optional[str] = field(default=None, metadata={"help": "Name"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_args_into_dataclasses(args=["--name", "test"])

        assert args.name == "test"

    def test_uses_default_value(self):
        """Test that default values are used when args not provided."""
        @dataclass
        class Args:
            name: str = field(default="default", metadata={"help": "Name"})
            count: int = field(default=5, metadata={"help": "Count"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_args_into_dataclasses(args=[])

        assert args.name == "default"
        assert args.count == 5

    def test_parses_multiple_dataclasses(self):
        """Test parsing multiple dataclass types."""
        @dataclass
        class Config1:
            name: str = field(default="config1", metadata={"help": "Name"})

        @dataclass
        class Config2:
            value: int = field(default=10, metadata={"help": "Value"})

        parser = HfArgumentParser((Config1, Config2))
        config1, config2 = parser.parse_args_into_dataclasses(
            args=["--name", "test", "--value", "20"]
        )

        assert config1.name == "test"
        assert config2.value == 20

    def test_parse_dict(self):
        """Test parsing from dictionary."""
        @dataclass
        class Args:
            name: str = field(default="default", metadata={"help": "Name"})
            count: int = field(default=5, metadata={"help": "Count"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_dict({"name": "from_dict", "count": 42})

        assert args.name == "from_dict"
        assert args.count == 42

    def test_parse_dict_ignores_extra_keys(self):
        """Test that parse_dict handles extra keys gracefully."""
        @dataclass
        class Args:
            name: str = field(default="default", metadata={"help": "Name"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_dict(
            {"name": "test", "extra_key": "ignored"},
            allow_extra_keys=True
        )

        assert args.name == "test"
        assert not hasattr(args, "extra_key")

    def test_handles_missing_metadata(self):
        """Test parsing fields without metadata."""
        @dataclass
        class Args:
            name: str = "default"

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_args_into_dataclasses(args=["--name", "test"])

        assert args.name == "test"

    def test_field_with_default_factory(self):
        """Test parsing fields with default_factory."""
        @dataclass
        class Args:
            items: list = field(default_factory=list, metadata={"help": "Items"})

        parser = HfArgumentParser((Args,))
        (args,) = parser.parse_args_into_dataclasses(args=[])

        assert args.items == []
        assert isinstance(args.items, list)
