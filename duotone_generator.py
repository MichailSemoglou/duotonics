"""Duotonics: pure duotone processing with GUI and CLI shells."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import os
import random
import re
import shutil
import stat
import struct
import tempfile
import warnings
from collections.abc import Iterator

import cv2
import numpy as np

__version__ = "1.0.0"

DEFAULT_VARIATION_COUNT = 100

# Extensions accepted by convert_to_duotone. Anything else is rejected before
# loading, so failures are reported against the advertised formats.
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Longest permitted image side in pixels. Images whose total pixel count
# exceeds MAX_IMAGE_DIMENSION ** 2 are rejected from the file header before
# any decode, so accidental gigapixel inputs cannot exhaust memory.
MAX_IMAGE_DIMENSION = 15000

# Blend spaces accepted by convert_to_duotone and create_duotone_image.
# 'srgb' preserves the pre-Duotonics implementation behavior exactly;
# 'linear' and 'oklab' blend over a relative-luminance key instead of the
# legacy luma key.
SPACES = ("srgb", "linear", "oklab")

# Maximum pixels processed together for the CIEDE2000 metric. The full
# difference plane is retained for statistics, but Lab arrays stay tile-sized.
_CIEDE2000_TILE_PIXELS = 1_000_000

# Fixed sRGB (IEC 61966-2-1) to XYZ matrix and D65 white point, used by the
# CIEDE2000 metric only.
_SRGB_TO_XYZ = np.array((
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
), dtype=np.float32)
_D65_WHITE = np.array((0.95047, 1.0, 1.08883), dtype=np.float32)


def _success_message(output_folder: str, count: int) -> str:
    """Build the completion message shared by the GUI and CLI shells."""
    noun = "variation" if count == 1 else "variations"
    return f"{count} duotone {noun} generated successfully in '{output_folder}'."


def select_image() -> None:
    """Open file dialogs to select a source image and output directory, then process."""
    from tkinter import filedialog, messagebox

    image_path = filedialog.askopenfilename(
        filetypes=[("Image files", " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS))]
    )
    if not image_path:
        return
    output_dir = filedialog.askdirectory(title="Select output directory")
    if not output_dir:
        return
    # Surface each conversion failure in the GUI rather than hiding it.
    try:
        output_folder, records = convert_to_duotone(image_path, output_dir=output_dir)
    except Exception as e:  # noqa: BLE001
        messagebox.showerror("Error", f"An error occurred: {e!s}")
        return
    messagebox.showinfo("Success", _success_message(output_folder, len(records)))


def generate_random_color(rng: random.Random | None = None) -> list[int]:
    """Generate a random RGB color, optionally from a seeded Random instance."""
    source = rng if rng is not None else random
    return [source.randint(0, 255) for _ in range(3)]


def rgb_to_hex(color: list[int]) -> str:
    """Convert an RGB list to a hex string."""
    return f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def hex_to_rgb(value: str) -> list[int]:
    """
    Convert a hex color string to an RGB list.

    Accepts '#RRGGBB' and 'RRGGBB', case-insensitively.
    Raises ValueError for anything else.
    """
    text = value.strip().removeprefix("#")
    if re.fullmatch(r"[0-9A-Fa-f]{6}", text) is None:
        raise ValueError(f"Invalid hex color '{value}': expected '#RRGGBB'.")
    return [int(text[i:i + 2], 16) for i in (0, 2, 4)]


def parse_color_pair(spec: str) -> tuple[list[int], list[int]]:
    """Parse a 'HEX1,HEX2' string into a pair of RGB lists."""
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid color pair '{spec}': expected 'HEX1,HEX2'.")
    return hex_to_rgb(parts[0]), hex_to_rgb(parts[1])


def _validate_color_pairs(colors: list[tuple[list[int], list[int]]]) -> None:
    """
    Validate explicit color pairs and reject duplicates.

    Raises ValueError if any color is not three integers in [0, 255], or if
    two pairs are identical: identical pairs map to identical filenames, so
    the second variation would silently overwrite the first.
    """
    seen = set()
    for pair in colors:
        try:
            pair_len = len(pair)
        except TypeError as exc:
            raise ValueError(
                f"Invalid color pair {pair!r}: expected a pair of two RGB colors."
            ) from exc
        if pair_len != 2:
            raise ValueError(
                f"Invalid color pair {pair!r}: expected exactly two RGB colors."
            )
        color1, color2 = pair
        for color in (color1, color2):
            try:
                color_len = len(color)
            except TypeError as exc:
                raise ValueError(
                    f"Invalid RGB color {color!r}: expected three integers in [0, 255]."
                ) from exc
            if color_len != 3 or not all(
                isinstance(v, (int, np.integer)) and 0 <= v <= 255 for v in color
            ):
                raise ValueError(
                    f"Invalid RGB color {color}: expected three integers "
                    "in [0, 255]."
                )
        hex_pair = rgb_to_hex(color1), rgb_to_hex(color2)
        if hex_pair in seen:
            raise ValueError(
                f"Duplicate color pair {hex_pair[0]},{hex_pair[1]}: "
                "each pair may appear only once."
            )
        seen.add(hex_pair)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """
    Linearize gamma-encoded sRGB code values in [0, 1].

    Applies the inverse EOTF of IEC 61966-2-1; see Poynton (2003),
    "Digital Video and HDTV". Returns an array of the same shape.

    Args:
        rgb: sRGB code values in [0, 1], any shape.

    Returns:
        Linear-light values with the input's shape, float32.
    """
    x = np.asarray(rgb, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Encode linear-light values as sRGB code values in [0, 1].

    Applies the forward EOTF of IEC 61966-2-1; see Poynton (2003).

    Args:
        linear: Linear-light values, treated as zero below zero.

    Returns:
        sRGB code values with the input's shape, float32.
    """
    v = np.maximum(np.asarray(linear, dtype=np.float32), 0.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert gamma-encoded sRGB in [0, 1] to Oklab (Ottosson, 2020).

    Oklab is a perceptually uniform space; interpolating in it avoids the
    midpoint desaturation and hue drift of gamma-space blending.

    Args:
        rgb: sRGB code values in [0, 1], shape (..., 3).

    Returns:
        Oklab values (L, a, b) with the input's shape, float32.
    """
    linear = _srgb_to_linear(rgb)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    lms_l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    lms_m = 0.2119034980 * r + 0.6806995451 * g + 0.1073969566 * b
    lms_s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    lms_l, lms_m, lms_s = np.cbrt(lms_l), np.cbrt(lms_m), np.cbrt(lms_s)
    return np.stack((
        0.2104542553 * lms_l + 0.7936177850 * lms_m - 0.0040720468 * lms_s,
        1.9779984951 * lms_l - 2.4285922050 * lms_m + 0.4505937099 * lms_s,
        0.0259040371 * lms_l + 0.7827717662 * lms_m - 0.8086757660 * lms_s,
    ), axis=-1).astype(np.float32)


def _oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert Oklab (Ottosson, 2020) to gamma-encoded sRGB in [0, 1].

    Args:
        lab: Oklab values (L, a, b), shape (..., 3).

    Returns:
        sRGB code values with the input's shape, float32. Values outside
        the sRGB gamut are negative or above 1; callers must clip.
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    lms_l, lms_m, lms_s = l_ ** 3, m_ ** 3, s_ ** 3
    rgb_linear = np.stack((
        4.0767416621 * lms_l - 3.3077115913 * lms_m + 0.2309699292 * lms_s,
        -1.2684380036 * lms_l + 2.6097574011 * lms_m - 0.3413193965 * lms_s,
        -0.0041960863 * lms_l - 0.7034186147 * lms_m + 1.7076147010 * lms_s,
    ), axis=-1)
    return _linear_to_srgb(rgb_linear).astype(np.float32)


def _duotone_rgb(
    key: np.ndarray,
    color1: list[int],
    color2: list[int],
    space: str,
) -> np.ndarray:
    """
    Blend color1 -> color2 over key, returning RGB in [0, 1].

    'srgb' interpolates gamma-encoded values (legacy behavior);
    'linear' interpolates linear light (IEC 61966-2-1; Poynton, 2003);
    'oklab' interpolates in Oklab (Ottosson, 2020) per channel and converts
    back to sRGB.

    Args:
        key: Blend weight as a float array normalized to [0, 1].
        color1: RGB color mapped to key 0, three integers in [0, 255].
        color2: RGB color mapped to key 1, three integers in [0, 255].
        space: One of SPACES.

    Returns:
        RGB image with the key's height and width in [0, 1], except for
        minor range drift from numeric error. The srgb path retains the
        key dtype for legacy rounding; newer spaces use float32. Callers
        clip.
    """
    c1 = np.array(color1, dtype=np.float32).reshape(1, 1, 3) / 255
    c2 = np.array(color2, dtype=np.float32).reshape(1, 1, 3) / 255
    w = np.asarray(key)[:, :, np.newaxis]
    if space == "srgb":
        return (1 - w) * c1 + w * c2
    w = w.astype(np.float32, copy=False)
    if space == "linear":
        return _linear_to_srgb(
            (1 - w) * _srgb_to_linear(c1) + w * _srgb_to_linear(c2)
        )
    o1, o2 = _rgb_to_oklab(c1), _rgb_to_oklab(c2)
    return np.clip(_oklab_to_rgb((1 - w) * o1 + w * o2), 0.0, 1.0)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert gamma-encoded sRGB in [0, 1] to CIE L*a*b* under D65.

    Uses the fixed sRGB-to-XYZ matrix and white point assumed by IEC
    61966-2-1. Used only as input to the CIEDE2000 metric.

    Args:
        rgb: sRGB code values in [0, 1], shape (..., 3).

    Returns:
        L*a*b* values with the input's shape, float32. L is in [0, 100];
        a and b are unbounded.
    """
    xyz = _srgb_to_linear(rgb) @ _SRGB_TO_XYZ.T
    ratio = xyz / _D65_WHITE
    delta = 6 / 29
    t = np.where(ratio > delta ** 3, np.cbrt(ratio),
                 ratio / (3 * delta ** 2) + 4 / 29)
    L = 116 * t[..., 1] - 16
    a = 500 * (t[..., 0] - t[..., 1])
    b = 200 * (t[..., 1] - t[..., 2])
    return np.stack((L, a, b), axis=-1).astype(np.float32)


def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    CIEDE2000 color difference between two arrays of L*a*b* values.

    Implements the reference formula with the corrections published by
    Sharma, Wu & Dalal (2005), "The CIEDE2000 Color-Difference Formula:
    Implementation Notes, Supplementary Test Data, and Observations".

    Args:
        lab1, lab2: L*a*b* arrays of the same shape (..., 3).

    Returns:
        ΔE00 values with the inputs' leading shape, float32.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(C_bar ** 7 / (C_bar ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p
    # Hue difference folded into (-180, 180] so averaging is continuous.
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh) / 2)

    # Mean hue h_bar; the case branches follow Sharma, Wu & Dalal (2005).
    h_diff = np.abs(h1p - h2p)
    h_sum = h1p + h2p
    h_bar = np.where(
        C1p * C2p == 0,
        h_sum,
        np.where(h_diff <= 180, h_sum / 2,
                 np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2)),
    )

    T = (1
         - 0.17 * np.cos(np.radians(h_bar - 30))
         + 0.24 * np.cos(np.radians(2 * h_bar))
         + 0.32 * np.cos(np.radians(3 * h_bar + 6))
         - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    L_bar = (L1 + L2) / 2
    S_L = 1 + (0.015 * (L_bar - 50) ** 2) / np.sqrt(20 + (L_bar - 50) ** 2)
    C_barp = (C1p + C2p) / 2
    S_C = 1 + 0.045 * C_barp
    S_H = 1 + 0.015 * C_barp * T
    d_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    R_C = 2 * np.sqrt(C_barp ** 7 / (C_barp ** 7 + 25 ** 7))
    R_T = -R_C * np.sin(np.radians(2 * d_theta))

    dE = ((dLp / S_L) ** 2
          + (dCp / S_C) ** 2
          + (dHp / S_H) ** 2
          + R_T * (dCp / S_C) * (dHp / S_H))
    return np.sqrt(np.maximum(dE, 0.0))


def _duotone_delta_e00_tile(
    key: np.ndarray,
    output_bgr: np.ndarray,
    color1: list[int],
    color2: list[int],
) -> np.ndarray:
    """Return CIEDE2000 differences for one bounded row tile."""
    output_rgb = output_bgr[:, :, ::-1].astype(np.float32) / 255
    reference_rgb = _duotone_rgb(key, color1, color2, "oklab")
    return _ciede2000(_rgb_to_lab(output_rgb), _rgb_to_lab(reference_rgb))


def create_duotone_image(
    gray: np.ndarray,
    color1: list[int],
    color2: list[int],
    space: str = "srgb",
) -> np.ndarray:
    """
    Create a duotone image from a blend key and two colors.

    Args:
        gray: Blend key as a float array normalized to [0, 1], with the
            output's height and width. Values come from convert_to_duotone:
            grayscale BT.601 luma for 'srgb' (legacy), or relative
            luminance for 'linear' and 'oklab'.
        color1: First RGB color, three integers in [0, 255]. Maps to key 0.
        color2: Second RGB color, three integers in [0, 255]. Maps to key 1.
        space: Where the interpolation runs: 'srgb' (default) blends
            gamma-encoded sRGB exactly as the pre-Duotonics implementation
            did; 'linear' blends linear light (IEC 61966-2-1; Poynton, 2003);
            'oklab' blends in Oklab (Ottosson, 2020).

    Returns:
        The duotone image as a uint8 array in OpenCV's BGR channel order,
        ready for cv2.imwrite.

    Raises:
        ValueError: If space is not one of 'srgb', 'linear', 'oklab'.
    """
    if space not in SPACES:
        raise ValueError(f"space must be one of {', '.join(SPACES)}, got '{space}'.")
    duotone = np.clip(_duotone_rgb(gray, color1, color2, space), 0.0, 1.0)
    duotone_rgb = np.clip(np.rint(duotone * 255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(duotone_rgb, cv2.COLOR_RGB2BGR)


_SOF_MARKERS = frozenset({
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
})
def _read_image_dimensions(image_path: str, ext: str) -> tuple[int, int]:
    """
    Return (width, height) by reading only the file header.

    Reads fixed-size PNG and BMP headers and scans JPEG marker headers
    incrementally. Raises ValueError on truncated or unrecognised headers.
    """
    try:
        with open(image_path, "rb") as fh:
            if ext == ".png":
                hdr = fh.read(24)
                if len(hdr) < 24 or hdr[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"'{image_path}' is not a valid PNG file.")
                width, height = struct.unpack(">II", hdr[16:24])
                return width, height

            if ext in (".jpg", ".jpeg"):
                if fh.read(2) != b"\xff\xd8":
                    raise ValueError(f"'{image_path}' is not a valid JPEG file.")
                while True:
                    prefix = fh.read(1)
                    if not prefix:
                        raise ValueError(
                            f"Truncated JPEG header in '{image_path}'."
                        )
                    if prefix != b"\xff":
                        raise ValueError(
                            f"Failed to read JPEG dimensions from '{image_path}'."
                        )
                    while prefix == b"\xff":
                        prefix = fh.read(1)
                        if not prefix:
                            raise ValueError(
                                f"Truncated JPEG header in '{image_path}'."
                            )
                    marker = prefix[0]
                    if marker in _SOF_MARKERS:
                        header = fh.read(7)
                        if len(header) < 7:
                            raise ValueError(
                                f"Truncated JPEG header in '{image_path}'."
                            )
                        return struct.unpack(">HH", header[3:7])
                    if marker in (0xD9, 0xDA):
                        raise ValueError(
                            f"Failed to read JPEG dimensions from '{image_path}'."
                        )
                    if 0xD0 <= marker <= 0xD7 or marker == 0x01:
                        continue
                    length_bytes = fh.read(2)
                    if len(length_bytes) < 2:
                        raise ValueError(
                            f"Truncated JPEG header in '{image_path}'."
                        )
                    segment_length = struct.unpack(">H", length_bytes)[0]
                    if segment_length < 2:
                        raise ValueError(
                            f"Invalid JPEG segment length in '{image_path}'."
                        )
                    fh.seek(segment_length - 2, 1)

            if ext == ".bmp":
                hdr = fh.read(26)
                if len(hdr) < 26 or hdr[:2] != b"BM":
                    raise ValueError(f"'{image_path}' is not a valid BMP file.")
                dib_size = struct.unpack("<I", hdr[14:18])[0]
                if dib_size == 12:
                    return struct.unpack("<HH", hdr[18:22])
                if dib_size >= 40:
                    width = abs(struct.unpack("<i", hdr[18:22])[0])
                    height = abs(struct.unpack("<i", hdr[22:26])[0])
                    return width, height
                raise ValueError(
                    f"Unsupported DIB header size {dib_size} in '{image_path}'."
                )

    except OSError as exc:
        raise ValueError(f"Failed to load image '{image_path}': {exc}") from exc
    except struct.error as exc:
        raise ValueError(
            f"Failed to read image dimensions from '{image_path}': {exc}"
        ) from exc

    raise ValueError(f"Unsupported extension '{ext}'.")  # unreachable for validated ext


def _load_image(image_path: str) -> np.ndarray:
    """
    Load an image as a BGR uint8 array.

    Raises ValueError for unsupported extensions, empty files, images whose
    total pixel count exceeds MAX_IMAGE_DIMENSION ** 2 (checked from the file
    header before any decode), and undecodable content.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext or '(none)'}' for "
            f"'{image_path}'. Supported formats: JPEG, PNG, BMP."
        )
    if os.path.isfile(image_path) and os.path.getsize(image_path) == 0:
        raise ValueError(
            f"Failed to load image '{image_path}': the file is empty (0 bytes)."
        )

    width, height = _read_image_dimensions(image_path, ext)
    if width * height > MAX_IMAGE_DIMENSION ** 2:
        raise ValueError(
            f"Image '{image_path}' is {width}x{height} pixels, exceeding the "
            f"maximum supported dimension of {MAX_IMAGE_DIMENSION} pixels."
        )

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image '{image_path}'.")
    return image


def _duotone_key(image: np.ndarray, space: str) -> np.ndarray:
    """
    Return the blend key in [0, 1] used by the given space.

    'srgb' keeps the legacy key, ITU-R BT.601 luma weights applied to
    gamma-encoded code values, so seeded batches remain byte-identical to
    the pre-Duotonics implementation. 'linear' and 'oklab' use relative
    luminance instead: ITU-R BT.709 weights applied to linearized RGB
    (IEC 61966-2-1; Poynton, 2003).

    Args:
        image: BGR uint8 image.
        space: One of SPACES.

    Returns:
        Float32 key array normalized to [0, 1], same height and width as
        the input.
    """
    if space == "srgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    rgb = image[:, :, ::-1].astype(np.float32) / 255
    linear = _srgb_to_linear(rgb)
    return (0.2126 * linear[..., 0]
            + 0.7152 * linear[..., 1]
            + 0.0722 * linear[..., 2])


def _open_regular_file(path: str) -> int:
    """Open a regular file without blocking or following symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Unable to open '{path}' safely: {exc}") from exc
    try:
        regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
    except OSError:
        os.close(descriptor)
        raise
    if not regular:
        os.close(descriptor)
        raise ValueError(f"Unsupported special file in output folder: '{path}'.")
    return descriptor


def _read_previous_manifest(output_folder: str) -> set[str]:
    """Return filenames owned by the previous batch manifest."""
    manifest_path = os.path.join(output_folder, "colors.csv")
    if not os.path.exists(manifest_path):
        return set()
    owned = {"colors.csv", "metrics.csv"}
    with os.fdopen(_open_regular_file(manifest_path), "r", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["variation", "color_1_hex", "color_2_hex"]:
            raise ValueError(f"Invalid batch manifest '{manifest_path}'.")
        for row in reader:
            try:
                index = int(row["variation"])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid batch manifest '{manifest_path}'.") from None
            if index < 0:
                raise ValueError(f"Invalid batch manifest '{manifest_path}'.")
            color1 = row["color_1_hex"].upper().removeprefix("#")
            color2 = row["color_2_hex"].upper().removeprefix("#")
            if any(
                re.fullmatch(r"[0-9A-F]{6}", color) is None
                for color in (color1, color2)
            ):
                raise ValueError(f"Invalid batch manifest '{manifest_path}'.")
            owned.add(f"{index:03d}_{color1}_{color2}.png")
    return owned


def _file_digest(path: str) -> str:
    """Hash one file's contents to detect same-size, same-mtime edits."""
    digest = hashlib.sha256()
    with os.fdopen(_open_regular_file(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unrelated_snapshot(
    output_folder: str, excluded: set[str]
) -> dict[str, tuple[str, int, str]]:
    """Capture unrelated file metadata to detect live-folder changes."""
    snapshot: dict[str, tuple[str, int, str]] = {}
    if not os.path.isdir(output_folder):
        return snapshot
    for directory, directories, filenames in os.walk(output_folder):
        for name in directories + filenames:
            path = os.path.join(directory, name)
            relative = os.path.relpath(path, output_folder)
            if relative in excluded:
                continue
            if os.path.islink(path):
                snapshot[relative] = ("link", 0, os.readlink(path))
            elif os.path.isdir(path):
                snapshot[relative] = (
                    "directory", os.lstat(path).st_mtime_ns, ""
                )
            else:
                snapshot[relative] = ("file", 0, _file_digest(path))
    return snapshot


def _assert_no_unowned_conflicts(
    output_folder: str, planned_files: set[str], previous_owned: set[str]
) -> None:
    """Reject planned output names that would overwrite unowned files."""
    if not os.path.isdir(output_folder):
        return
    for name in planned_files - previous_owned:
        path = os.path.join(output_folder, name)
        if os.path.exists(path):
            raise ValueError(
                f"Planned output '{path}' conflicts with an unowned file."
            )


@contextlib.contextmanager
def _batch_output_folder(
    output_folder: str, base: str, stem: str, planned_files: set[str]
) -> Iterator[str]:
    """Stage a complete batch and publish it with a directory swap."""
    os.makedirs(base or ".", exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=f".{stem}_duotone_", dir=base or ".")
    preserve_workdir = False
    published = False
    try:
        previous_owned = (
            _read_previous_manifest(output_folder)
            if os.path.isdir(output_folder)
            else set()
        )
        _assert_no_unowned_conflicts(output_folder, planned_files, previous_owned)
        excluded = previous_owned | planned_files
        snapshot = _unrelated_snapshot(output_folder, excluded)
        next_folder = os.path.join(workdir, "next")
        if os.path.exists(output_folder):
            if not os.path.isdir(output_folder):
                raise NotADirectoryError(
                    f"Output path '{output_folder}' is not a directory."
                )

            def ignore_previous_batch(directory: str, names: list[str]) -> set[str]:
                if os.path.abspath(directory) != os.path.abspath(output_folder):
                    return set()
                return {name for name in names if name in previous_owned}

            shutil.copytree(
                output_folder,
                next_folder,
                symlinks=True,
                ignore=ignore_previous_batch,
            )
        else:
            os.makedirs(next_folder)

        yield next_folder

        _assert_no_unowned_conflicts(output_folder, planned_files, previous_owned)
        if _unrelated_snapshot(output_folder, excluded) != snapshot:
            raise RuntimeError(
                "Output folder changed during generation; previous output was "
                "left unchanged."
            )

        previous_folder = os.path.join(workdir, "previous")
        try:
            if os.path.exists(output_folder):
                os.replace(output_folder, previous_folder)
            os.replace(next_folder, output_folder)
            published = True
        except (OSError, KeyboardInterrupt):
            if os.path.exists(previous_folder) and not os.path.exists(output_folder):
                try:
                    os.replace(previous_folder, output_folder)
                except OSError as rollback_error:
                    preserve_workdir = True
                    raise OSError(
                        f"Failed to publish '{output_folder}'; the previous batch "
                        f"is retained at '{previous_folder}'."
                    ) from rollback_error
            raise
    finally:
        if not preserve_workdir:
            if published:
                try:
                    shutil.rmtree(workdir)
                except OSError as cleanup_error:
                    warnings.warn(
                        f"Batch published at '{output_folder}', but temporary files "
                        f"remain at '{workdir}': {cleanup_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                shutil.rmtree(workdir)


def convert_to_duotone(
    image_path: str,
    output_dir: str | None = None,
    count: int = DEFAULT_VARIATION_COUNT,
    seed: int | None = None,
    colors: list[tuple[list[int], list[int]]] | None = None,
    space: str = "srgb",
) -> tuple[str, list[list[int | str]]]:
    """
    Convert an image to duotone variations and save them to disk.

    For each variation a color pair is blended over the source's tonal key
    and written as a lossless PNG. Filenames encode both hex color values
    (e.g. '083_2CD2B4_DC1E5A.png'). A 'colors.csv' sidecar lists every
    variation's color pair, and a 'metrics.csv' sidecar reports the mean,
    median, and 95th-percentile CIEDE2000 (Sharma, Wu & Dalal, 2005)
    between each output and its perceptual reference: the same color pair
    blended in Oklab (Ottosson, 2020) over the same key.

    Args:
        image_path: Absolute or relative path to the source image.
            Supported formats: JPEG, PNG, BMP. Loading uses cv2.imread
            with IMREAD_COLOR, so 16-bit images are downconverted to
            8-bit and alpha channels are dropped.
        output_dir: Directory under which the
            '<image_stem>_duotone_variations' folder is created. When None
            (default) the folder is placed alongside the source image.
        count: Number of random variations to generate
            (default: DEFAULT_VARIATION_COUNT). Ignored when `colors` is given.
        seed: Seed for random color generation. Pass an integer to reproduce
            a batch exactly; None (default) produces a different batch
            every run.
        colors: Explicit color pairs as (color1, color2) RGB lists, one
            variation per pair. When given, `count` and `seed` are unused.
        space: Where the blend runs: 'srgb' (default) interpolates
            gamma-encoded sRGB over the legacy grayscale key, byte-identical
            to the pre-Duotonics implementation; 'linear' and 'oklab'
            interpolate in linear sRGB (IEC 61966-2-1; Poynton, 2003) and
            Oklab (Ottosson, 2020) over an ITU-R BT.709 relative-luminance
            key.

    Returns:
        A tuple ``(output_folder, color_records)``: the path of the created
        '<image_stem>_duotone_variations' folder and a list of
        ``[index, color_1_hex, color_2_hex]`` rows, one per variation.

    Raises:
        ValueError: If `image_path` has an unsupported extension (supported
            formats: JPEG, PNG, BMP), the file is empty (0 bytes), the image
            cannot be decoded, or its total pixel count exceeds
            MAX_IMAGE_DIMENSION ** 2 (checked from the file header before
            decoding). Also if `count` is below 1, `colors` is an empty list,
            an explicit color is not three integers in [0, 255], two color
            pairs are identical, or `space` is not one of 'srgb', 'linear',
            'oklab'.
        OSError: If a variation file cannot be written to disk.

    Side effects:
        Creates '<output_dir>/<image_stem>_duotone_variations/' (or a sibling
        folder when output_dir is None) containing one PNG per variation,
        one 'colors.csv' file, and one 'metrics.csv' file. The batch is written
        to a temporary sibling directory before publication. Existing generated
        files listed in the previous batch manifest are replaced, and unrelated
        files are preserved. If generation fails or unrelated files change during
        generation, the previous output remains unchanged.
    """
    if space not in SPACES:
        raise ValueError(f"space must be one of {', '.join(SPACES)}, got '{space}'.")
    if colors is None:
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}.")
    elif not colors:
        raise ValueError("colors must contain at least one color pair.")
    else:
        _validate_color_pairs(colors)

    key = _duotone_key(_load_image(image_path), space)

    if colors is None:
        # Seeded colors support reproducible art direction, not security.
        rng = random.Random(seed)  # nosec B311
        pairs = [
            (generate_random_color(rng), generate_random_color(rng))
            for _ in range(count)
        ]
    else:
        pairs = list(colors)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    base = output_dir if output_dir is not None else os.path.dirname(image_path)
    output_folder = os.path.join(base, f"{stem}_duotone_variations")

    planned_files = {
        f"{i:03d}_{rgb_to_hex(color1).lstrip('#')}_"
        f"{rgb_to_hex(color2).lstrip('#')}.png"
        for i, (color1, color2) in enumerate(pairs)
    }
    with _batch_output_folder(
        output_folder, base, stem, planned_files
    ) as staged_folder:
        color_records: list[list[int | str]] = []
        metric_records: list[list[int | float | str]] = []
        for i, (color1, color2) in enumerate(pairs):
            duotone = create_duotone_image(key, color1, color2, space)

            hex1 = rgb_to_hex(color1)
            hex2 = rgb_to_hex(color2)

            filename = f"{i:03d}_{hex1.lstrip('#')}_{hex2.lstrip('#')}.png"
            output_path = os.path.join(staged_folder, filename)
            if not cv2.imwrite(output_path, duotone):
                raise OSError(f"Failed to write '{output_path}'.")
            color_records.append([i, hex1, hex2])

            # Per-pixel CIEDE2000 (Sharma, Wu & Dalal, 2005) between this
            # variation and its perceptual reference: the same pair blended
            # in Oklab over the same key. In the 'oklab' space this measures
            # only quantization error; in the other spaces it measures the
            # perceptual cost of that blend space.
            height, width = key.shape
            tile_rows = max(1, _CIEDE2000_TILE_PIXELS // width)
            delta_e = np.empty((height, width), dtype=np.float32)
            for row_start in range(0, height, tile_rows):
                row_end = min(row_start + tile_rows, height)
                delta_e[row_start:row_end] = _duotone_delta_e00_tile(
                    key[row_start:row_end],
                    duotone[row_start:row_end],
                    color1,
                    color2,
                )
            metric_records.append([
                i, hex1, hex2, space,
                round(float(delta_e.mean()), 4),
                round(float(np.median(delta_e)), 4),
                round(float(np.percentile(delta_e, 95)), 4),
            ])

        csv_path = os.path.join(staged_folder, "colors.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["variation", "color_1_hex", "color_2_hex"])
            writer.writerows(color_records)

        metrics_path = os.path.join(staged_folder, "metrics.csv")
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["variation", "color_1_hex", "color_2_hex", "space",
                 "mean_deltaE00", "median_deltaE00", "p95_deltaE00"]
            )
            writer.writerows(metric_records)

    return output_folder, color_records


def main(argv: list[str] | None = None) -> None:
    """Console entry point: process the image argument, or launch the GUI."""
    parser = argparse.ArgumentParser(
        description="Generate duotone variations of an image."
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to the source image (JPEG, PNG, BMP). "
             "When omitted, file dialogs open instead.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory under which the output folder is created "
             "(default: alongside the source image).",
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=None,
        help=f"Number of random variations to generate "
             f"(default: {DEFAULT_VARIATION_COUNT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for random color generation; reuse a seed to reproduce "
             "a batch exactly.",
    )
    parser.add_argument(
        "--colors",
        action="append",
        default=None,
        metavar="HEX1,HEX2",
        help="Explicit color pair(s) instead of random ones, e.g. "
             "--colors '#2CD2B4,#DC1E5A'. Repeatable; one variation per pair. "
             "Cannot be combined with --count or --seed.",
    )
    parser.add_argument(
        "--space",
        choices=SPACES,
        default=None,
        help="Color space for the blend: srgb is the legacy gamma-space "
             "blend; linear and oklab blend in linear sRGB and Oklab "
             "(Ottosson, 2020) over a relative-luminance key "
             "(default: srgb).",
    )
    args = parser.parse_args(argv)

    if not args.image:
        flags_used = (
            args.output_dir is not None
            or args.count is not None
            or args.seed is not None
            or args.colors
            or args.space is not None
        )
        if flags_used:
            parser.error(
                "--output-dir, --count, --seed, --colors and --space require "
                "an image path"
            )
        try:
            select_image()
        except ImportError:
            parser.exit(1, "Error: tkinter is not installed; "
                           "pass an image path to use CLI mode.\n")
        return

    space = args.space or "srgb"
    colors = None
    if args.colors:
        if args.count is not None or args.seed is not None:
            parser.error("--colors cannot be combined with --count or --seed")
        try:
            colors = [parse_color_pair(spec) for spec in args.colors]
        except ValueError as e:
            parser.error(str(e))

    count = args.count if args.count is not None else DEFAULT_VARIATION_COUNT

    # Surface each processing failure as a concise CLI error with a nonzero exit.
    try:
        output_folder, records = convert_to_duotone(
            args.image, output_dir=args.output_dir, count=count,
            seed=args.seed, colors=colors, space=space,
        )
    except Exception as e:  # noqa: BLE001
        parser.exit(1, f"Error: {e}\n")

    print(_success_message(output_folder, len(records)))


if __name__ == "__main__":
    main()
