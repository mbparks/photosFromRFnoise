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
"""

import argparse
import json
import logging
import sys
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
    p = argparse.ArgumentParser(
        description="Capture RF from an RTL-SDR and render it as borderless art.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Capture
    p.add_argument("-d", "--duration", type=float, default=2.0,
                   help="Capture duration in seconds.")
    p.add_argument("-s", "--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE,
                   help="Sample rate in Hz.")
    p.add_argument("-f", "--freq", type=float, default=None,
                   help="Center frequency in Hz. If omitted, picked at random.")
    p.add_argument("--freq-min", type=float, default=DEFAULT_FREQ_MIN_HZ,
                   help="Lower bound for random frequency selection (Hz).")
    p.add_argument("--freq-max", type=float, default=DEFAULT_FREQ_MAX_HZ,
                   help="Upper bound for random frequency selection (Hz).")
    p.add_argument("-g", "--gain", default="auto",
                   help="Tuner gain in dB, or 'auto'.")
    p.add_argument("--ppm", type=int, default=0,
                   help="Frequency correction in parts per million.")
    # Rendering
    p.add_argument("-m", "--mode", default="spectrogram",
                   choices=("spectrogram", "constellation", "polar"),
                   help="Visualization style.")
    p.add_argument("--cmap", default="viridis",
                   help="Matplotlib colormap (viridis, magma, inferno, twilight, "
                        "turbo, plasma, cividis, hsv, ...).")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE,
                   help="Output image size in pixels (square).")
    p.add_argument("--nfft", type=int, default=1024,
                   help="FFT size for spectrogram mode.")
    # Filters
    p.add_argument("--filters", type=int, default=3,
                   help="Number of random filters to apply (0 = none).")
    p.add_argument("--filter-list", default=None,
                   help="Comma-separated explicit filter names "
                        "(e.g. 'bloom,chromatic_shift,kaleidoscope'). "
                        "Overrides --filters when provided.")
    p.add_argument("--filter-seed", type=int, default=None,
                   help="Seed for filter selection / parameters.")
    p.add_argument("--list-filters", action="store_true",
                   help="Print the available filter names and exit.")
    # Output
    p.add_argument("--output-dir", type=Path, default=Path("./sdr_runs"),
                   help="Parent directory; each run creates a timestamped subfolder.")
    p.add_argument("--csv", action="store_true",
                   help="Also write IQ samples as iq.csv in the run folder.")
    p.add_argument("--csv-decimate", type=int, default=1,
                   help="Keep every Nth IQ sample in the CSV.")
    # Misc
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for random frequency selection.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug logging.")
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
                   output_path, folder) -> None:
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
        "csv_written": bool(args.csv),
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

    if args.csv:
        write_csv(iq, actual_rate, folder / "iq.csv", decimate=args.csv_decimate)

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
                   output_path, folder)

    logging.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
