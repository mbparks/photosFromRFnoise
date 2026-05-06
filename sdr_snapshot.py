#!/usr/bin/env python3
"""
sdr_snapshot.py — Tune an RTL-SDR to a (random) frequency, capture IQ samples,
and render the result as a borderless PNG art piece with optional random filters.

Every run lands in its own date-time-stamped folder under --output-dir.

Dependencies:
    pip install pyrtlsdr numpy matplotlib pillow

You also need the librtlsdr system library:
    Debian/Ubuntu : sudo apt install librtlsdr-dev rtl-sdr
    Fedora        : sudo dnf install rtl-sdr-devel
    macOS         : brew install librtlsdr
    Windows       : install Zadig drivers and put rtlsdr.dll on the PATH

Examples:
    # Spectrogram with default 3 random filters
    python sdr_snapshot.py

    # Polar mode, 5 filters, twilight cyclic colormap
    python sdr_snapshot.py --mode polar --cmap twilight --filters 5

    # Constellation, no filters, also dump CSV
    python sdr_snapshot.py --mode constellation --filters 0 --csv

    # Reproducible run (same freq, same filter chain) at 4K
    python sdr_snapshot.py --seed 42 --filter-seed 42 --size 4096 --filters 4

    # Apply specific filters in a specific order
    python sdr_snapshot.py --filter-list bloom,chromatic_shift,kaleidoscope,vignette

    # Full export bundle for AI workflows: image + IQ WAV + FM-demod audio +
    # features.json + iq.csv
    python sdr_snapshot.py --export-all

    # Just the AI-friendly bits (no CSV)
    python sdr_snapshot.py --wav --demod fm --features
"""

import argparse
import json
import logging
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib as mpl
from PIL import Image, ImageFilter

try:
    from rtlsdr import RtlSdr
except ImportError as e:
    sys.exit(
        f"Failed to import pyrtlsdr: {e}\n\n"
        "This usually means one of two things:\n"
        "  1. The Python package is missing:  pip install pyrtlsdr\n"
        "  2. The librtlsdr C library is missing or not on the loader path:\n"
        "       macOS    : brew install librtlsdr\n"
        "       Debian   : sudo apt install librtlsdr-dev rtl-sdr\n"
        "       Fedora   : sudo dnf install rtl-sdr-devel\n"
        "       Windows  : install Zadig drivers and put rtlsdr.dll on PATH"
    )


# Practical tuning range for the common R820T / R820T2 tuners.
DEFAULT_FREQ_MIN_HZ = 24e6
DEFAULT_FREQ_MAX_HZ = 1_700e6

# 2.048 MHz is a clean, well-supported sample rate for RTL-SDR.
DEFAULT_SAMPLE_RATE = 2.048e6

# librtlsdr requires read sizes that are multiples of 256.
CHUNK_SAMPLES = 256 * 1024

DEFAULT_SIZE = 1024


# =========================================================================== #
# CLI
# =========================================================================== #

def parse_args():
    epilog = """\
examples:
  # Default: random freq, 2s capture, spectrogram, 3 random filters
  sdr_snapshot.py

  # Polar mode with cyclic colormap and lots of filters
  sdr_snapshot.py --mode polar --cmap twilight --filters 6

  # Lock to FM broadcast band
  sdr_snapshot.py --freq-min 88e6 --freq-max 108e6 -g 40

  # Reproduce an exact prior result
  sdr_snapshot.py --seed 7 --filter-seed 13 --mode polar --cmap twilight

  # Specific filter chain in order
  sdr_snapshot.py --filter-list bloom,chromatic_shift,kaleidoscope,vignette

  # Full export bundle for AI workflows: image + IQ WAV + demod audio +
  # features.json + iq.csv
  sdr_snapshot.py --export-all

  # Just the AI-friendly bits, no filters (clean image for ControlNet)
  sdr_snapshot.py --filters 0 --wav --demod fm --features

  # List all available filters and exit
  sdr_snapshot.py --list-filters

See README.md for the full reference, troubleshooting, and AI workflow recipes.
"""
    p = argparse.ArgumentParser(
        description="Capture RF from an RTL-SDR and render it as borderless art.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    capture = p.add_argument_group("capture", "How the radio captures samples")
    capture.add_argument("-d", "--duration", type=float, default=2.0, metavar="SEC",
                         help="Capture duration in seconds (default: 2.0)")
    capture.add_argument("-s", "--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE,
                         metavar="HZ",
                         help="Sample rate in Hz (default: 2.048e6)")
    capture.add_argument("-f", "--freq", type=float, default=None, metavar="HZ",
                         help="Center frequency in Hz. If omitted, picked at random.")
    capture.add_argument("--freq-min", type=float, default=DEFAULT_FREQ_MIN_HZ,
                         metavar="HZ",
                         help="Lower bound for random freq selection (default: 24e6)")
    capture.add_argument("--freq-max", type=float, default=DEFAULT_FREQ_MAX_HZ,
                         metavar="HZ",
                         help="Upper bound for random freq selection (default: 1.7e9)")
    capture.add_argument("-g", "--gain", default="auto", metavar="DB",
                         help="Tuner gain in dB, or 'auto' for AGC (default: auto)")
    capture.add_argument("--ppm", type=int, default=0,
                         help="Frequency correction in parts per million (default: 0)")

    render = p.add_argument_group("rendering", "How the image is generated")
    render.add_argument("-m", "--mode", default="spectrogram",
                        choices=("spectrogram", "constellation", "polar"),
                        help="Visualization style (default: spectrogram)")
    render.add_argument("--cmap", default="viridis", metavar="NAME",
                        help="Matplotlib colormap: viridis, magma, inferno, twilight, "
                             "turbo, plasma, cividis, hsv, ... (default: viridis)")
    render.add_argument("--size", type=int, default=DEFAULT_SIZE, metavar="PX",
                        help="Output image size in pixels, square (default: 1024)")
    render.add_argument("--nfft", type=int, default=1024,
                        help="FFT size for spectrogram mode (default: 1024)")

    filters = p.add_argument_group("filters", "Post-processing applied to the image")
    filters.add_argument("--filters", type=int, default=3, metavar="N",
                         help="Number of random filters to apply, 0 disables (default: 3)")
    filters.add_argument("--filter-list", default=None, metavar="NAMES",
                         help="Comma-separated explicit filter names. "
                              "Overrides --filters when provided.")
    filters.add_argument("--filter-seed", type=int, default=None, metavar="N",
                         help="Seed for filter selection and parameters")
    filters.add_argument("--list-filters", action="store_true",
                         help="Print available filter names and exit")

    output = p.add_argument_group("output", "Where files go and what gets written")
    output.add_argument("--output-dir", type=Path, default=Path("./sdr_runs"),
                        metavar="DIR",
                        help="Parent dir; each run gets a timestamped subfolder "
                             "(default: ./sdr_runs)")
    output.add_argument("--csv", action="store_true",
                        help="Write IQ samples to iq.csv")
    output.add_argument("--csv-decimate", type=int, default=1, metavar="N",
                        help="Keep every Nth IQ sample in the CSV (default: 1)")
    output.add_argument("--wav", action="store_true",
                        help="Write decimated IQ as stereo iq.wav (I=L, Q=R)")
    output.add_argument("--wav-rate", type=int, default=48000, metavar="HZ",
                        help="Target sample rate for WAV exports (default: 48000)")
    output.add_argument("--demod", default="none", choices=("none", "fm", "am"),
                        help="Demodulate IQ and write audio_demod_<mode>.wav "
                             "(default: none)")
    output.add_argument("--features", action="store_true",
                        help="Write features.json with signal characteristics "
                             "(useful for AI prompts)")
    output.add_argument("--export-all", action="store_true",
                        help="Shorthand for --csv --wav --demod fm --features")

    misc = p.add_argument_group("misc")
    misc.add_argument("--seed", type=int, default=None, metavar="N",
                      help="Seed for random frequency selection")
    misc.add_argument("-v", "--verbose", action="store_true",
                      help="Enable debug logging")

    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


# =========================================================================== #
# SDR setup & capture
# =========================================================================== #

def pick_frequency(args) -> float:
    if args.freq is not None:
        return float(args.freq)
    rng = np.random.default_rng(args.seed)
    if args.freq_min >= args.freq_max:
        raise ValueError("--freq-min must be less than --freq-max")
    return float(rng.uniform(args.freq_min, args.freq_max))


def open_sdr(center_freq: float, sample_rate: float, gain, ppm: int) -> RtlSdr:
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq

    if ppm:
        try:
            sdr.freq_correction = ppm
        except OSError as e:
            logging.warning("Could not set ppm correction (%s); continuing.", e)

    if isinstance(gain, str) and gain.lower() == "auto":
        try:
            sdr.gain = "auto"
        except (TypeError, ValueError):
            sdr.set_manual_gain_enabled(False)
    else:
        sdr.gain = float(gain)

    return sdr


def capture_iq(sdr: RtlSdr, duration: float) -> np.ndarray:
    sample_rate = sdr.sample_rate
    total_samples = int(round(duration * sample_rate))
    total_samples = ((total_samples + 255) // 256) * 256

    logging.info(
        "Capturing %.3f s -> %d samples at %.3f Msps",
        duration, total_samples, sample_rate / 1e6,
    )

    chunks = []
    remaining = total_samples
    while remaining > 0:
        n = min(CHUNK_SAMPLES, remaining)
        n = ((n + 255) // 256) * 256
        chunks.append(sdr.read_samples(n))
        remaining -= n

    return np.concatenate(chunks)[:total_samples]


# =========================================================================== #
# Data → image  (no axes, no labels, no chrome)
# =========================================================================== #

def apply_colormap(data: np.ndarray, cmap_name: str,
                   vmin: float = None, vmax: float = None) -> np.ndarray:
    """2D float array -> (H, W, 3) uint8 RGB via matplotlib colormap."""
    if vmin is None:
        vmin = float(np.percentile(data, 1))
    if vmax is None:
        vmax = float(np.percentile(data, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = mpl.colormaps[cmap_name]
    rgba = cmap(norm)  # H x W x 4 floats in [0, 1]
    return (rgba[..., :3] * 255).astype(np.uint8)


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img).resize((size, size), Image.LANCZOS)
    return np.array(pil)


def make_spectrogram_image(iq: np.ndarray, nfft: int,
                            cmap: str, size: int) -> np.ndarray:
    hop = nfft // 2
    window = np.hanning(nfft).astype(np.float32)

    n_frames = max(1, (len(iq) - nfft) // hop + 1)
    spec = np.empty((n_frames, nfft), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = iq[start:start + nfft] * window
        spec[i] = np.fft.fftshift(np.abs(np.fft.fft(frame)))

    spec_db = 20.0 * np.log10(spec + 1e-12)
    rgb = apply_colormap(spec_db, cmap)
    return resize_square(rgb, size)


def make_constellation_image(iq: np.ndarray, cmap: str, size: int) -> np.ndarray:
    i, q = iq.real, iq.imag
    lim = float(np.percentile(np.abs(iq), 99.5))
    h, _, _ = np.histogram2d(
        i, q, bins=size,
        range=[[-lim, lim], [-lim, lim]],
    )
    # Transpose so I→x, Q→y, then flipud so Q increases upward.
    img2d = np.flipud(np.log1p(h).T)
    return apply_colormap(img2d, cmap)


def make_polar_image(iq: np.ndarray, cmap: str, size: int,
                     phase_bins: int = 720, mag_bins: int = 256) -> np.ndarray:
    phase = np.angle(iq)
    mag = np.abs(iq)
    mag_lim = float(np.percentile(mag, 99.5))

    hist, _, _ = np.histogram2d(
        phase, mag,
        bins=(phase_bins, mag_bins),
        range=[[-np.pi, np.pi], [0.0, mag_lim]],
    )
    hist_log = np.log1p(hist)

    # Cartesian → polar lookup: corners get the outermost magnitude bin
    # (so the image fills the frame instead of being a circle on a square).
    cx = cy = (size - 1) / 2.0
    radius = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)  # -pi..pi

    p_idx = ((angle + np.pi) / (2 * np.pi) * phase_bins).astype(np.int32) % phase_bins
    m_idx = np.clip((r / radius * mag_bins).astype(np.int32), 0, mag_bins - 1)

    polar = hist_log[p_idx, m_idx]
    return apply_colormap(polar, cmap)


# =========================================================================== #
# Filter suite — every filter takes (img: HxWx3 uint8, rng) and returns same.
# =========================================================================== #

def filter_chromatic_shift(img, rng):
    """RGB channel offsets — VHS-style chromatic aberration."""
    max_shift = max(8, img.shape[1] // 50)
    out = img.copy()
    for ch in (0, 2):
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        out[..., ch] = np.roll(img[..., ch], (dy, dx), axis=(0, 1))
    return out


def filter_slice_shift(img, rng):
    """Datamosh: split into horizontal bands, shift each one randomly."""
    h, w = img.shape[:2]
    n_slices = int(rng.integers(8, 30))
    edges = np.sort(rng.integers(0, h, n_slices))
    edges = np.concatenate(([0], edges, [h]))
    max_shift = w // 6
    out = img.copy()
    for i in range(len(edges) - 1):
        y0, y1 = int(edges[i]), int(edges[i + 1])
        if y1 <= y0:
            continue
        shift = int(rng.integers(-max_shift, max_shift + 1))
        out[y0:y1] = np.roll(img[y0:y1], shift, axis=1)
    return out


def filter_hue_rotate(img, rng):
    """Rotate hue by a random amount via HSV roundtrip."""
    pil = Image.fromarray(img).convert("HSV")
    hsv = np.array(pil)
    shift = int(rng.integers(0, 256))
    hsv[..., 0] = (hsv[..., 0].astype(np.int32) + shift) % 256
    return np.array(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


def filter_invert(img, rng):
    """Hard color invert."""
    return 255 - img


def filter_solarize(img, rng):
    """Invert pixels above a random threshold."""
    threshold = int(rng.integers(64, 192))
    out = img.copy()
    mask = img > threshold
    out[mask] = 255 - img[mask]
    return out


def filter_posterize(img, rng):
    """Quantize each channel to a few discrete levels."""
    levels = int(rng.integers(2, 8))
    step = 256 // levels
    return ((img.astype(np.int32) // step) * step).clip(0, 255).astype(np.uint8)


def filter_gaussian_blur(img, rng):
    """Soft Gaussian blur."""
    radius = float(rng.uniform(1.5, 6.0))
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def filter_pixelate(img, rng):
    """Chunky downsample-then-nearest-upsample."""
    h, w = img.shape[:2]
    factor = int(rng.integers(8, 40))
    pil = Image.fromarray(img)
    small = pil.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return np.array(small.resize((w, h), Image.NEAREST))


def filter_wave_distort(img, rng):
    """Sine-wave displacement of rows or columns."""
    h, w = img.shape[:2]
    amp = float(rng.uniform(8, 40))
    freq = float(rng.uniform(0.005, 0.04))
    axis = int(rng.integers(0, 2))
    out = np.empty_like(img)
    if axis == 0:
        shifts = (amp * np.sin(2 * np.pi * freq * np.arange(h))).astype(int)
        for y in range(h):
            out[y] = np.roll(img[y], int(shifts[y]), axis=0)
    else:
        shifts = (amp * np.sin(2 * np.pi * freq * np.arange(w))).astype(int)
        for x in range(w):
            out[:, x] = np.roll(img[:, x], int(shifts[x]), axis=0)
    return out


def filter_mirror_h(img, rng):
    """Mirror across the horizontal centerline."""
    h = img.shape[0]
    out = img.copy()
    if rng.integers(0, 2):
        out[h // 2:] = img[:h - h // 2][::-1]
    else:
        n = h // 2
        out[:n] = img[h - n:][::-1]
    return out


def filter_mirror_v(img, rng):
    """Mirror across the vertical centerline."""
    w = img.shape[1]
    out = img.copy()
    if rng.integers(0, 2):
        out[:, w // 2:] = img[:, :w - w // 2][:, ::-1]
    else:
        n = w // 2
        out[:, :n] = img[:, w - n:][:, ::-1]
    return out


def filter_kaleidoscope(img, rng):
    """4-way kaleidoscope from the top-left quadrant."""
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    quad = img[:h2, :w2]
    out = img.copy()
    out[:h2, :w2] = quad
    out[:h2, w - w2:] = quad[:, ::-1]
    out[h - h2:, :w2] = quad[::-1, :]
    out[h - h2:, w - w2:] = quad[::-1, ::-1]
    return out


def filter_bloom(img, rng):
    """Glow: extract highlights, blur big, additively composite back."""
    threshold = int(rng.integers(150, 220))
    radius = float(rng.uniform(10, 25))
    intensity = float(rng.uniform(0.4, 1.0))

    highlights = img.astype(np.float32)
    highlights[img < threshold] = 0
    blurred = np.array(
        Image.fromarray(np.clip(highlights, 0, 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(radius=radius))
    ).astype(np.float32)

    return np.clip(img.astype(np.float32) + intensity * blurred, 0, 255).astype(np.uint8)


def filter_noise(img, rng):
    """Add Gaussian film grain."""
    amount = float(rng.uniform(8, 30))
    noise = rng.normal(0, amount, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def filter_vignette(img, rng):
    """Soft radial darkening (gradient — not a hard frame)."""
    h, w = img.shape[:2]
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    strength = float(rng.uniform(0.3, 0.8))
    falloff = float(rng.uniform(1.5, 3.0))
    mask = np.clip(1 - strength * np.power(radius, falloff), 0, 1)
    return (img.astype(np.float32) * mask[..., None]).astype(np.uint8)


def filter_channel_swap(img, rng):
    """Permute the RGB channels."""
    perm = list(range(3))
    rng.shuffle(perm)
    return img[..., perm]


FILTERS = {
    "chromatic_shift": filter_chromatic_shift,
    "slice_shift":     filter_slice_shift,
    "hue_rotate":      filter_hue_rotate,
    "invert":          filter_invert,
    "solarize":        filter_solarize,
    "posterize":       filter_posterize,
    "gaussian_blur":   filter_gaussian_blur,
    "pixelate":        filter_pixelate,
    "wave_distort":    filter_wave_distort,
    "mirror_h":        filter_mirror_h,
    "mirror_v":        filter_mirror_v,
    "kaleidoscope":    filter_kaleidoscope,
    "bloom":           filter_bloom,
    "noise":           filter_noise,
    "vignette":        filter_vignette,
    "channel_swap":    filter_channel_swap,
}


def pick_random_filters(n: int, rng) -> list:
    if n <= 0:
        return []
    keys = list(FILTERS.keys())
    n = min(n, len(keys))
    return [str(x) for x in rng.choice(keys, size=n, replace=False)]


def apply_filter_chain(img: np.ndarray, names: list, rng) -> np.ndarray:
    for name in names:
        if name not in FILTERS:
            logging.warning("Unknown filter '%s' — skipping.", name)
            continue
        logging.info("  + %s", name)
        img = FILTERS[name](img, rng)
    return img


# =========================================================================== #
# CSV / metadata / folder
# =========================================================================== #

def lowpass_decimate(data: np.ndarray, factor: int, num_taps: int = 64) -> np.ndarray:
    """Anti-alias FIR filter then decimate. Works for real or complex input."""
    if factor <= 1:
        return data
    cutoff = 0.5 / factor
    taps = np.sinc(2 * cutoff * (np.arange(num_taps) - (num_taps - 1) / 2.0))
    taps *= np.hamming(num_taps)
    taps /= np.sum(taps)

    if np.iscomplexobj(data):
        re = np.convolve(data.real, taps, mode="same")
        im = np.convolve(data.imag, taps, mode="same")
        filtered = re + 1j * im
    else:
        filtered = np.convolve(data, taps, mode="same")
    return filtered[::factor]


def _to_int16(samples: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """Scale a real-valued array to fill the int16 range."""
    peak = float(np.max(np.abs(samples)))
    if peak == 0.0:
        return np.zeros_like(samples, dtype=np.int16)
    scale = headroom * 32767.0 / peak
    return np.clip(samples * scale, -32768, 32767).astype(np.int16)


def write_wav(path: Path, samples_int16: np.ndarray, sample_rate: int,
              n_channels: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        wf.writeframes(samples_int16.tobytes())


def write_iq_wav(iq: np.ndarray, sample_rate: float, path: Path,
                 target_rate: int = 48000) -> None:
    """Decimate IQ to target_rate and write as stereo 16-bit WAV (I=L, Q=R)."""
    factor = max(1, int(round(sample_rate / target_rate)))
    actual_rate = int(round(sample_rate / factor))
    iq_dec = lowpass_decimate(iq, factor)

    # Joint scaling so I and Q stay in correct relative amplitude
    peak = float(max(np.max(np.abs(iq_dec.real)), np.max(np.abs(iq_dec.imag))))
    scale = 0.95 * 32767.0 / peak if peak > 0 else 1.0

    interleaved = np.empty(2 * len(iq_dec), dtype=np.int16)
    interleaved[0::2] = np.clip(iq_dec.real * scale, -32768, 32767).astype(np.int16)
    interleaved[1::2] = np.clip(iq_dec.imag * scale, -32768, 32767).astype(np.int16)

    write_wav(path, interleaved, actual_rate, n_channels=2)
    size_mb = path.stat().st_size / (1024 * 1024)
    logging.info("Wrote %s  (stereo IQ, %.1f kHz, %.1f MB)",
                 path, actual_rate / 1000, size_mb)


def demod_fm(iq: np.ndarray) -> np.ndarray:
    """Quadrature FM demodulation. Returns instantaneous frequency in radians/sample."""
    # Phase difference between consecutive samples; equivalent to angle of
    # iq[1:] * conj(iq[:-1]) but avoids the intermediate complex array.
    return np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)


def demod_am(iq: np.ndarray) -> np.ndarray:
    """Envelope (AM) demodulation. Returns DC-removed magnitude."""
    env = np.abs(iq).astype(np.float32)
    return env - np.mean(env)


def write_audio_wav(iq: np.ndarray, sample_rate: float, path: Path,
                    mode: str, target_rate: int = 48000) -> None:
    """Demodulate and write mono audio at ~target_rate."""
    if mode == "fm":
        audio = demod_fm(iq)
    elif mode == "am":
        audio = demod_am(iq)
    else:
        raise ValueError(f"Unknown demod mode: {mode}")

    factor = max(1, int(round(sample_rate / target_rate)))
    actual_rate = int(round(sample_rate / factor))
    audio_dec = lowpass_decimate(audio, factor)

    write_wav(path, _to_int16(audio_dec), actual_rate, n_channels=1)
    size_mb = path.stat().st_size / (1024 * 1024)
    logging.info("Wrote %s  (%s demod, mono, %.1f kHz, %.1f MB)",
                 path, mode.upper(), actual_rate / 1000, size_mb)


def compute_features(iq: np.ndarray, sample_rate: float,
                     center_freq: float, nfft: int = 4096) -> dict:
    """Extract human-meaningful signal characteristics from IQ data."""
    # ---- Amplitude domain ----
    env = np.abs(iq)
    env_mean = float(np.mean(env))
    env_std = float(np.std(env))
    rms = float(np.sqrt(np.mean(env * env)))
    peak = float(np.max(env))
    crest_db = 20.0 * np.log10(peak / rms) if rms > 0 else 0.0

    # Burstiness: how much does the envelope vary across short windows?
    win = max(1, len(env) // 64)
    if win > 1:
        chunks = env[:len(env) // win * win].reshape(-1, win)
        chunk_means = chunks.mean(axis=1)
        burstiness = float(np.std(chunk_means) / (np.mean(chunk_means) + 1e-12))
    else:
        burstiness = 0.0

    # ---- Frequency domain (averaged PSD) ----
    nfft = min(nfft, len(iq))
    hop = nfft // 2
    n_frames = max(1, (len(iq) - nfft) // hop + 1)
    window = np.hanning(nfft).astype(np.float32)
    psd = np.zeros(nfft, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        frame = iq[s:s + nfft] * window
        psd += np.abs(np.fft.fftshift(np.fft.fft(frame))) ** 2
    psd /= n_frames
    psd_db = 10.0 * np.log10(psd + 1e-20)
    freqs_hz = np.linspace(-sample_rate / 2, sample_rate / 2, nfft)

    noise_floor_db = float(np.percentile(psd_db, 10))
    peak_db = float(np.max(psd_db))
    peak_to_noise_db = peak_db - noise_floor_db

    # Occupied bandwidth: freq range containing X% of the total power
    psd_norm = psd / (np.sum(psd) + 1e-20)
    cdf = np.cumsum(psd_norm)
    def occupied_bw(fraction):
        lo = float(freqs_hz[np.searchsorted(cdf, (1 - fraction) / 2)])
        hi = float(freqs_hz[min(len(cdf) - 1,
                                 np.searchsorted(cdf, 1 - (1 - fraction) / 2))])
        return hi - lo
    bw_99 = occupied_bw(0.99)
    bw_90 = occupied_bw(0.90)

    # Spectral entropy normalized to [0, 1]; 1 = uniform / noise-like
    spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-20)) / np.log2(nfft))

    # Top peaks (offset from center, in kHz)
    smoothed = np.convolve(psd_db, np.ones(7) / 7.0, mode="same")
    n_peaks = 5
    peak_indices = []
    work = smoothed.copy()
    min_separation = nfft // 64
    for _ in range(n_peaks):
        idx = int(np.argmax(work))
        if work[idx] < noise_floor_db + 6:
            break
        peak_indices.append(idx)
        lo = max(0, idx - min_separation)
        hi = min(nfft, idx + min_separation)
        work[lo:hi] = -np.inf
    dominant_peaks_khz = [round(float(freqs_hz[i]) / 1000.0, 3)
                          for i in peak_indices]

    # ---- Instantaneous frequency stats (FM-bandwidth proxy) ----
    inst_freq_hz = np.diff(np.unwrap(np.angle(iq))) * sample_rate / (2 * np.pi)
    inst_freq_mean_khz = float(np.mean(inst_freq_hz)) / 1000.0
    inst_freq_std_khz = float(np.std(inst_freq_hz)) / 1000.0

    # ---- Plain-English summary string ----
    if peak_to_noise_db > 25:
        strength = "strong"
    elif peak_to_noise_db > 12:
        strength = "moderate"
    elif peak_to_noise_db > 6:
        strength = "faint"
    else:
        strength = "no clear"

    if bw_99 < 25e3:
        bw_word = "narrowband"
    elif bw_99 < 250e3:
        bw_word = "medium-bandwidth"
    else:
        bw_word = "wideband"

    envelope_word = "bursty" if burstiness > 0.4 else "continuous"
    n_carriers = len(dominant_peaks_khz)
    carriers_word = (
        f"{n_carriers} dominant carrier{'s' if n_carriers != 1 else ''}"
        if n_carriers > 0 else "no clear carriers"
    )

    summary = (
        f"{strength} {bw_word} signal at {center_freq/1e6:.3f} MHz with "
        f"{envelope_word} envelope and {carriers_word}"
    )

    return {
        "center_freq_mhz": round(center_freq / 1e6, 6),
        "sample_rate_msps": round(sample_rate / 1e6, 6),
        "duration_s": round(len(iq) / sample_rate, 4),
        "amplitude": {
            "mean": env_mean,
            "std": env_std,
            "rms": rms,
            "peak": peak,
            "crest_factor_db": round(crest_db, 2),
            "envelope_burstiness": round(burstiness, 4),
        },
        "spectrum": {
            "noise_floor_db": round(noise_floor_db, 2),
            "peak_db": round(peak_db, 2),
            "peak_to_noise_db": round(peak_to_noise_db, 2),
            "occupied_bandwidth_99pct_khz": round(bw_99 / 1000.0, 2),
            "occupied_bandwidth_90pct_khz": round(bw_90 / 1000.0, 2),
            "spectral_entropy_normalized": round(spec_entropy, 4),
            "dominant_peaks_offset_khz": dominant_peaks_khz,
        },
        "phase": {
            "instantaneous_freq_mean_khz": round(inst_freq_mean_khz, 3),
            "instantaneous_freq_std_khz": round(inst_freq_std_khz, 3),
        },
        "summary": summary,
    }


def write_features_json(features: dict, path: Path) -> None:
    path.write_text(json.dumps(features, indent=2))
    logging.info("Wrote %s", path)
    logging.info("  Summary: %s", features["summary"])


def write_csv(iq: np.ndarray, sample_rate: float, path: Path,
              decimate: int = 1) -> None:
    if decimate < 1:
        raise ValueError("--csv-decimate must be >= 1")
    iq_out = iq[::decimate]
    n = len(iq_out)
    sample_idx = np.arange(n, dtype=np.int64) * decimate
    time_s = sample_idx / sample_rate

    data = np.column_stack([
        sample_idx, time_s,
        iq_out.real, iq_out.imag,
        np.abs(iq_out), np.angle(iq_out),
    ])
    np.savetxt(
        path, data, delimiter=",",
        header="sample_index,time_s,i,q,magnitude,phase", comments="",
        fmt=["%d", "%.9e", "%.6e", "%.6e", "%.6e", "%.6e"],
    )
    size_mb = path.stat().st_size / (1024 * 1024)
    logging.info("Wrote %s  (%d rows, %.1f MB)", path, n, size_mb)


def write_metadata(args, center_freq, sample_rate, filters_applied,
                   output_path, folder, exports) -> None:
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "center_freq_hz": float(center_freq),
        "center_freq_mhz": float(center_freq) / 1e6,
        "sample_rate_hz": float(sample_rate),
        "duration_s": float(args.duration),
        "gain": str(args.gain),
        "ppm": int(args.ppm),
        "cmap": args.cmap,
        "size_px": int(args.size),
        "nfft": int(args.nfft),
        "filters_applied": list(filters_applied),
        "filter_seed": args.filter_seed,
        "freq_seed": args.seed,
        "image_file": output_path.name,
        "exports": exports,
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))


def create_run_folder(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = base_dir / f"sdr_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# =========================================================================== #
# Main
# =========================================================================== #

def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if args.list_filters:
        print("Available filters:")
        for name in FILTERS:
            print(f"  {name}")
        sys.exit(0)

    folder = create_run_folder(args.output_dir)
    logging.info("Run folder: %s", folder)

    center_freq = pick_frequency(args)
    logging.info("Center frequency: %.6f MHz", center_freq / 1e6)
    logging.info("Sample rate:      %.6f MHz", args.sample_rate / 1e6)
    logging.info("Duration:         %.3f s", args.duration)
    logging.info("Mode:             %s   cmap=%s   size=%d", args.mode, args.cmap, args.size)

    sdr = None
    try:
        sdr = open_sdr(center_freq, args.sample_rate, args.gain, args.ppm)
        actual_freq = sdr.center_freq
        actual_rate = sdr.sample_rate
        logging.debug("Device tuned: %.6f MHz @ %.6f Msps",
                      actual_freq / 1e6, actual_rate / 1e6)
        iq = capture_iq(sdr, args.duration)
    except (OSError, IOError) as e:
        logging.error("Could not talk to RTL-SDR: %s", e)
        logging.error("Device plugged in? Drivers loaded? "
                      "Another app holding it (SDR#, GQRX, rtl_fm, ...)?")
        sys.exit(1)
    finally:
        if sdr is not None:
            sdr.close()

    # Resolve --export-all shortcut into individual flags
    if args.export_all:
        args.csv = True
        args.wav = True
        args.features = True
        if args.demod == "none":
            args.demod = "fm"

    exports = {"csv": False, "iq_wav": False, "audio_wav": False, "features": False}

    if args.csv:
        write_csv(iq, actual_rate, folder / "iq.csv", decimate=args.csv_decimate)
        exports["csv"] = "iq.csv"

    if args.wav:
        write_iq_wav(iq, actual_rate, folder / "iq.wav", target_rate=args.wav_rate)
        exports["iq_wav"] = "iq.wav"

    if args.demod != "none":
        audio_path = folder / f"audio_demod_{args.demod}.wav"
        write_audio_wav(iq, actual_rate, audio_path, args.demod,
                        target_rate=args.wav_rate)
        exports["audio_wav"] = audio_path.name

    if args.features:
        feats = compute_features(iq, actual_rate, actual_freq, nfft=args.nfft)
        write_features_json(feats, folder / "features.json")
        exports["features"] = "features.json"

    logging.info("Generating %s image...", args.mode)
    if args.mode == "spectrogram":
        img = make_spectrogram_image(iq, args.nfft, args.cmap, args.size)
    elif args.mode == "constellation":
        img = make_constellation_image(iq, args.cmap, args.size)
    elif args.mode == "polar":
        img = make_polar_image(iq, args.cmap, args.size)

    rng = np.random.default_rng(args.filter_seed)
    if args.filter_list:
        filter_names = [n.strip() for n in args.filter_list.split(",") if n.strip()]
    else:
        filter_names = pick_random_filters(args.filters, rng)

    if filter_names:
        logging.info("Applying %d filter(s):", len(filter_names))
        img = apply_filter_chain(img, filter_names, rng)
    else:
        logging.info("No filters applied.")

    output_path = folder / f"{args.mode}_{actual_freq/1e6:.3f}MHz.png"
    Image.fromarray(img).save(output_path, "PNG")

    write_metadata(args, actual_freq, actual_rate, filter_names,
                   output_path, folder, exports)

    logging.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
