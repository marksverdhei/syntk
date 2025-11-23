"""Argument parser utilities for syntk.

Provides a fallback HfArgumentParser implementation when transformers is not available.
"""

try:
    from transformers import HfArgumentParser
except ImportError:
    # Fallback to argparse if transformers is not installed
    import dataclasses
    from argparse import ArgumentParser
    from typing import get_args, get_origin

    def _get_base_type(field_type):
        """Extract base type from Optional or direct type."""
        origin = get_origin(field_type)
        if origin is not None:  # Union type (Optional is Union[T, None])
            args = get_args(field_type)
            # Get the non-None type from Optional
            return next((arg for arg in args if arg is not type(None)), None)
        return field_type

    class HfArgumentParser:
        """Minimal replacement for HfArgumentParser when transformers is not available."""

        def __init__(self, dataclass_types):
            self.dataclass_types = dataclass_types
            self.parser = ArgumentParser()

            for dataclass_type in dataclass_types:
                for field in dataclasses.fields(dataclass_type):
                    field_name = f"--{field.name}"
                    field_metadata = field.metadata or {}
                    help_text = field_metadata.get("help", "")

                    kwargs = {"help": help_text}
                    if field.default is not dataclasses.MISSING:
                        kwargs["default"] = field.default
                    elif field.default_factory is not dataclasses.MISSING:
                        kwargs["default"] = field.default_factory()
                    else:
                        kwargs["required"] = True

                    base_type = _get_base_type(field.type)
                    if base_type is str:
                        kwargs["type"] = str
                    elif base_type is int:
                        kwargs["type"] = int
                    elif base_type is float:
                        kwargs["type"] = float

                    self.parser.add_argument(field_name, **kwargs)

        def parse_args_into_dataclasses(self, args=None):
            namespace = self.parser.parse_args(args)
            result = []
            for dataclass_type in self.dataclass_types:
                field_names = {f.name for f in dataclasses.fields(dataclass_type)}
                kwargs = {k: v for k, v in vars(namespace).items() if k in field_names}
                result.append(dataclass_type(**kwargs))
            return tuple(result)

        def parse_dict(self, config_dict, allow_extra_keys=False):
            result = []
            for dataclass_type in self.dataclass_types:
                field_names = {f.name for f in dataclasses.fields(dataclass_type)}
                kwargs = {k: v for k, v in config_dict.items() if k in field_names}
                result.append(dataclass_type(**kwargs))
            return tuple(result)
