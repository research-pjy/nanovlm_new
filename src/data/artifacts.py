"""Atomic publication for deterministic data artifacts; never replace differing files."""

import os
from pathlib import Path
import tempfile


def publish(output, payload):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.artifact-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Publish a complete file atomically, refusing to replace an existing file.
            os.link(temporary, output)
            status = 'created'
        except FileExistsError:
            if output.read_bytes() != payload:
                raise ValueError(f'Existing export differs: {output}; use a new output path')
            status = 'unchanged'
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return status
