"""PNG integrity and bounded pixel inspection without changing source images."""

import hashlib
import io
import struct
import warnings

import numpy as np
from PIL import Image


def inspect_image(path, decode=False):
    record = {"filestem": path.stem, "bytes": path.stat().st_size}
    try:
        # Read one compressed image at a time, reusing its bytes for CRC and hashing.
        data = path.read_bytes()
        record["sha256"] = hashlib.sha256(data).hexdigest()
        if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise ValueError("Invalid PNG header")
        width, height, depth, color, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", data[16:29]
        )
        record.update(width=width, height=height, bit_depth=depth, color_type=color,
                      compression=compression, filtering=filtering, interlace=interlace)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Image.open(io.BytesIO(data)) as image:
                record["mode"] = image.mode
                image.verify()
            record["warnings"] = [str(w.message) for w in caught]
        if decode:
            # Decoding checks are explicit; verify() alone does not prove pixel decoding.
            with Image.open(io.BytesIO(data)) as image:
                pixels = np.asarray(image)
                record["pixels"] = {
                    "dtype": str(pixels.dtype), "shape": list(pixels.shape),
                    "min": int(pixels.min()), "max": int(pixels.max()),
                    "mean": float(pixels.mean()),
                    "p01": float(np.quantile(pixels, 0.01)),
                    "p50": float(np.quantile(pixels, 0.5)),
                    "p99": float(np.quantile(pixels, 0.99)),
                    "zero_fraction": float(np.mean(pixels == 0)),
                    "maximum_code_fraction": float(np.mean(pixels == (2 ** depth - 1))),
                }
                if pixels.min() == pixels.max():
                    record["constant_pixels"] = True
    except (OSError, ValueError, SyntaxError, struct.error) as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
    return record


def distribution(values):
    if not values:
        return {"n": 0}
    data = np.asarray(values, dtype=float)
    return dict(zip(("min", "p01", "p25", "median", "p75", "p99", "max"),
                    map(float, np.quantile(data, [0, .01, .25, .5, .75, .99, 1]))),
                n=len(values), mean=float(data.mean()))
