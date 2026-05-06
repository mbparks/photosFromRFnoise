# sdr_snapshot

Tune an RTL-SDR dongle to a (random) frequency, capture IQ samples, and render the result as a borderless PNG art piece. Three visualization modes, sixteen funk-cranking filters, and a clean per-run output folder.

This started as a "spectrogram from a random frequency" toy and grew into a small generative-art tool. The radio is doing the random number generation.

---

## What it does

Each run:

1. Opens an RTL-SDR USB dongle.
2. Tunes it to a frequency you specify, or picks one at random within a configurable range.
3. Captures IQ samples for a configurable duration.
4. Renders those samples in one of three modes — `spectrogram`, `constellation`, or `polar`.
5. Optionally chains a random subset of 16 filters over the image (or a chain you specify).
6. Drops the PNG, an optional CSV of the raw IQ, and a `metadata.json` into a date-time-stamped folder.

The output PNGs have **no axes, no labels, no colorbars, no titles, no borders** — the image fills the entire frame.

---

## Hardware

Any RTL-SDR USB dongle with an R820T / R820T2 / R828D tuner. Defaults assume that family's tuning range (24 MHz – 1.7 GHz). Other tuners with different ranges can be accommodated with `--freq-min` / `--freq-max`.

---

## Install

### 1. Python dependencies

```bash
python3 -m venv ~/sdr-env
source ~/sdr-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install 'pyrtlsdr<0.3' 'setuptools<82' numpy matplotlib pillow
```

The version pins are deliberate — see [Troubleshooting](#troubleshooting). On Python ≤ 3.11 with an older librtlsdr you can usually skip them, but on a modern stack they save real pain.

### 2. The librtlsdr C library

The Python package `pyrtlsdr` is just a ctypes wrapper. You also need the C library on your system.

| OS              | Command                                       |
|-----------------|-----------------------------------------------|
| macOS           | `brew install librtlsdr`                      |
| Debian / Ubuntu | `sudo apt install librtlsdr-dev rtl-sdr`      |
| Fedora          | `sudo dnf install rtl-sdr-devel`              |
| Windows         | Install Zadig drivers, put `rtlsdr.dll` on `PATH` |

### 3. macOS only — point the loader at Homebrew

macOS doesn't search Homebrew library paths by default, and `pyrtlsdr<0.3` doesn't search them either, so you need to tell the dynamic loader where the dylib lives:

```bash
echo 'export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc
```

Use `/usr/local/lib` instead on Intel Macs. (`brew --prefix` will tell you which.)

### 4. Verify

```bash
python3 -c "from rtlsdr import RtlSdr; print('OK')"
```

If that prints `OK`, you're ready to run the script.

---

## Quick start

```bash
python3 sdr_snapshot.py
```

Tunes to a random frequency, captures 2 seconds, renders a spectrogram, applies 3 random filters, drops everything in `./sdr_runs/sdr_<timestamp>/`.

---

## Daily use

Virtual environments don't persist across terminal sessions, so every new shell needs the venv reactivated before running the script:

```bash
source ~/sdr-env/bin/activate
python3 sdr_snapshot.py
```

The prompt changes to `(sdr-env) ...` once it's active. Forgetting this step is the most common cause of "it worked yesterday, now it's broken" errors — see [Troubleshooting](#troubleshooting).

To make this less repetitive, add one of these to `~/.zshrc`:

```bash
# Option A — short alias to activate
alias sdr='source ~/sdr-env/bin/activate'

# Option B — one-shot wrapper that doesn't need activation at all
alias sdrshot='~/sdr-env/bin/python3 ~/path/to/sdr_snapshot.py'
```

With option B, `sdrshot --mode polar --cmap twilight` works from any directory in any terminal, no activation step required. That's the one I'd recommend.

---

## Modes

Pick one with `-m / --mode`:

| Mode            | What it shows                                                                  | Best for                              |
|-----------------|--------------------------------------------------------------------------------|---------------------------------------|
| `spectrogram`   | Time × frequency power, Hann-windowed FFTs with 50% overlap                    | Default; signal-rich frequencies      |
| `constellation` | Log-density 2D histogram of I vs Q in the complex plane                        | Modulated signals (FM/PSK/QPSK)       |
| `polar`         | Phase-vs-magnitude density, polar-projected to fill the frame corner-to-corner | Anything with a phase pattern         |

All three render at `--size × --size` pixels (default 1024).

---

## Filter suite

Sixteen post-processing filters can be applied in random order over the rendered image. Each filter pulls its own parameters from the `--filter-seed` RNG.

| Filter            | Effect                                                              |
|-------------------|---------------------------------------------------------------------|
| `chromatic_shift` | Independent RGB channel offsets — VHS-style chromatic aberration    |
| `slice_shift`     | Datamosh: split into horizontal bands, shift each one randomly      |
| `hue_rotate`      | Rotate the entire image's hue by a random amount                    |
| `invert`          | Hard color inversion                                                |
| `solarize`        | Invert pixels above a threshold                                     |
| `posterize`       | Quantize each channel to a small number of discrete levels          |
| `gaussian_blur`   | Soft Gaussian blur                                                  |
| `pixelate`        | Chunky downsample-then-nearest-upsample                             |
| `wave_distort`    | Sine-wave displacement of rows or columns                           |
| `mirror_h`        | Mirror across the horizontal centerline                             |
| `mirror_v`        | Mirror across the vertical centerline                               |
| `kaleidoscope`    | 4-way kaleidoscope from the top-left quadrant                       |
| `bloom`           | Glow effect — extract highlights, blur big, additively composite    |
| `noise`           | Gaussian film grain                                                 |
| `vignette`        | Soft radial darkening (gradient, not a hard frame)                  |
| `channel_swap`    | Permute the RGB channels                                            |

`--filters N` picks N at random. `--filter-list a,b,c` runs a specific chain. `--list-filters` prints the available names.

---

## CLI reference

| Flag                    | Default                  | Description                                                  |
|-------------------------|--------------------------|--------------------------------------------------------------|
| `-d, --duration`        | `2.0`                    | Capture duration (seconds)                                   |
| `-s, --sample-rate`     | `2.048e6`                | Sample rate (Hz)                                             |
| `-f, --freq`            | (random)                 | Center frequency (Hz); skips the random pick                 |
| `--freq-min`            | `24e6`                   | Lower bound for random frequency selection (Hz)              |
| `--freq-max`            | `1.7e9`                  | Upper bound for random frequency selection (Hz)              |
| `-g, --gain`            | `auto`                   | Tuner gain in dB, or `auto` for AGC                          |
| `--ppm`                 | `0`                      | Frequency correction in parts per million                    |
| `-m, --mode`            | `spectrogram`            | `spectrogram`, `constellation`, or `polar`                   |
| `--cmap`                | `viridis`                | Matplotlib colormap name                                     |
| `--size`                | `1024`                   | Output image size (square, in pixels)                        |
| `--nfft`                | `1024`                   | FFT size for spectrogram mode                                |
| `--filters`             | `3`                      | Number of random filters to apply (`0` to disable)           |
| `--filter-list`         | —                        | Comma-separated explicit filter names; overrides `--filters` |
| `--filter-seed`         | (random)                 | RNG seed for filter selection / parameters                   |
| `--list-filters`        | —                        | Print available filter names and exit                        |
| `--output-dir`          | `./sdr_runs`             | Parent directory for run folders                             |
| `--csv`                 | off                      | Also write IQ samples to `iq.csv`                            |
| `--csv-decimate`        | `1`                      | Keep every Nth IQ sample in the CSV                          |
| `--seed`                | (random)                 | RNG seed for random frequency selection                      |
| `-v, --verbose`         | off                      | Debug logging                                                |

---

## Output structure

```
./sdr_runs/
└── sdr_20260505_143022/
    ├── spectrogram_446.123MHz.png   ← the art
    ├── iq.csv                       ← raw IQ, only if --csv
    └── metadata.json                ← every parameter + filter chain
```

`metadata.json` records the frequency, sample rate, mode, colormap, filter chain, and seeds — so when one of the random outputs is great you can reproduce or tweak it precisely.

The CSV columns are `sample_index, time_s, i, q, magnitude, phase`. A 2-second capture at 2.048 Msps without decimation is roughly 200 MB; `--csv-decimate 100` brings that to ~2 MB.

---

## Recipes

```bash
# Maximum funk — cyclic colormap, polar mode, lots of filters
python3 sdr_snapshot.py --mode polar --cmap twilight --filters 6

# Same capture, five different filter chains (great for comparing looks)
for i in 1 2 3 4 5; do
  python3 sdr_snapshot.py --seed 42 --filter-seed $i \
    --mode constellation --cmap magma
done

# 4K print
python3 sdr_snapshot.py --size 4096 --filters 5 --cmap inferno

# Lock to FM broadcast band, capture longer, write CSV
python3 sdr_snapshot.py -d 5 --freq-min 88e6 --freq-max 108e6 \
  -g 40 --csv --csv-decimate 50

# Apply a specific filter chain in a specific order
python3 sdr_snapshot.py --filter-list bloom,chromatic_shift,kaleidoscope,vignette

# Reproduce an exact prior result
python3 sdr_snapshot.py --seed 7 --filter-seed 13 --mode polar --cmap twilight
```

### Colormaps that work especially well

- **`twilight`** — cyclic; pairs naturally with `polar` mode where the phase wraps
- **`magma`**, **`inferno`** — high contrast, work great with `bloom`
- **`turbo`** — punchy and saturated, good for `spectrogram`
- **`hsv`** — chaotic and rainbow-y, lean into it with `posterize`

### Frequency bands that produce more interesting captures than empty UHF

| Range            | Content                                |
|------------------|----------------------------------------|
| 88–108 MHz       | FM broadcast — bright structured rings in constellation mode |
| 118–137 MHz      | Aviation AM                            |
| 162–174 MHz      | NOAA weather, marine VHF               |
| 420–450 MHz      | 70 cm ham + PMR446                     |
| 433.05–434.79 MHz| ISM band, lots of remote controls / sensors |
| 902–928 MHz      | ISM band, LoRa, key fobs               |
| 1090 MHz         | ADS-B aircraft transponders            |

Empty noise looks fine too — it just renders as a Gaussian blob in constellation mode and a smooth ring in polar.

---

## Troubleshooting

The five problems most likely to hit you, in roughly the order people hit them:

### "It worked before — now it doesn't" after closing and reopening Terminal

Almost always: **the venv isn't active**. Your prompt should start with `(sdr-env)`. If it doesn't:

```bash
source ~/sdr-env/bin/activate
```

Without the venv active, `python3` resolves to the system framework Python, which doesn't have the pinned `setuptools<82` and `pyrtlsdr<0.3` — so you'll see weird import errors (`No module named 'pkg_resources'` is the most common). See [Daily use](#daily-use) for aliases that make this a one-time concern.

### `Error loading librtlsdr. Make sure librtlsdr ... are in your path`

The C library isn't installed, or the loader can't find it.

```bash
brew install librtlsdr
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix librtlsdr)/lib:$DYLD_FALLBACK_LIBRARY_PATH"
```

`DYLD_FALLBACK_LIBRARY_PATH` is the right knob on modern macOS — `DYLD_LIBRARY_PATH` gets stripped by SIP for some binaries. Persist the export in `~/.zshrc` to avoid setting it every shell.

### `dlsym(...) symbol not found: rtlsdr_set_dithering`

`pyrtlsdr` 0.3+ binds `rtlsdr_set_dithering` unconditionally at import. That symbol only exists in `librtlsdr` ≥ 2.0.

Either upgrade librtlsdr (`brew upgrade librtlsdr`, then verify with `nm -gU $(brew --prefix librtlsdr)/lib/librtlsdr.dylib | grep dithering`), or pin the wrapper:

```bash
python3 -m pip install 'pyrtlsdr<0.3'
```

### `ModuleNotFoundError: No module named 'pkg_resources'`

Two possible causes — check them in this order:

1. **Venv not active** — see the very first troubleshooting entry above. This is by far the most common cause on return visits.
2. **Setuptools 82+ in the venv** — `setuptools` 82 (released February 2026) finally removed `pkg_resources` after years of deprecation, and `pyrtlsdr<0.3` still uses it. Pin it down:

   ```bash
   python3 -m pip install 'setuptools<82'
   ```

### `pyrtlsdr is not installed` despite `pip` saying it is

This message can also appear when the C library isn't loadable. The script's import error message in older versions conflated the two. The current version surfaces the actual underlying exception — if you see this, you're running an old copy. Re-download `sdr_snapshot.py` and try again.

### Other miscellany

- **`Could not talk to RTL-SDR`** — usually another app (SDR#, GQRX, Cubic SDR, `rtl_fm`) is holding the device. Close it and try again.
- **`pip` warnings about `~yrtlsdr` directories** — pip's marker for files it couldn't fully clean up during uninstall. Cosmetic; safe to delete the `~`-prefixed directories manually with `sudo rm -rf` if they bother you.
- **Output looks flat / monotone** — the frequency you tuned to has no signal, or the filter chain happened to wash everything out. Try another `--seed`, another mode, fewer filters, or a known-active band from the table above.

---

## How it works (quick mental model)

The RTL-SDR streams 8-bit I and Q values at the configured sample rate; `pyrtlsdr` exposes them as `complex64` numpy arrays. From there:

- **Spectrogram** is a sliding-window FFT — each row is one Hann-windowed FFT magnitude in dB, 50% overlap.
- **Constellation** is a 2D histogram of `(real, imag)` pairs with log scaling so the dense noise core doesn't drown out the structure.
- **Polar** is a 2D histogram of `(phase, magnitude)` in polar coordinates, then remapped to a Cartesian grid via per-pixel `atan2` / radius lookup. The corners of the frame map to the outermost magnitude bin so the image fills the rectangle instead of being a circle on a square background.

All three produce a 2D float array. That array gets normalized between robust percentiles, pushed through a matplotlib colormap to get RGB, resized to `--size × --size` via PIL, and then any chosen filters are applied in sequence. Final save is a plain PIL `Image.save` with no figure-level decorations involved.

---

## License

Do whatever you want with it.
