#!/usr/bin/env python3
"""Run this TensorBoard with a local protobuf JSON compatibility adapter.

TensorBoard 2.16 passes the removed ``including_default_value_fields`` argument
when serving Custom Scalars. Current protobuf calls the equivalent option
``always_print_fields_with_no_presence``. This process-local adapter changes no
installed package, training process, event or metric.
"""
import functools
import inspect

from google.protobuf import json_format


def install_compatibility():
    original = json_format.MessageToJson
    parameters = inspect.signature(original).parameters
    if 'including_default_value_fields' in parameters:
        return False
    if 'always_print_fields_with_no_presence' not in parameters:
        raise RuntimeError('Unsupported protobuf MessageToJson signature')

    @functools.wraps(original)
    def compatible(message, *args, **kwargs):
        if 'including_default_value_fields' in kwargs:
            kwargs.setdefault('always_print_fields_with_no_presence',
                              kwargs.pop('including_default_value_fields'))
        return original(message, *args, **kwargs)

    json_format.MessageToJson = compatible
    return True


def main():
    install_compatibility()
    from tensorboard.main import run_main
    run_main()


if __name__ == '__main__':
    main()
