# syncs the govee lights to whatever the pc is playing
# python music.py

import asyncio
import colorsys
import math
import random
import threading
import time
import tkinter as tk
import warnings
from dataclasses import dataclass
from tkinter import colorchooser

import aubio
import numpy as np
import soundcard as sc

import lumisync_audio as lumi

from govee import ADDRESS, BULBS, KEEP_ALIVE, color, connect, power, send

FPS = 20
OFF = (0, 0, 0)
FLASH_DECAY = 0.15
BREATHE_LEAD = 0.2      # breathe fades slow so it runs 0.2s ahead -> looks on beat
LATE_OK = 0.05          # beats later than this get skipped, not sent late
DEFAULT_PERIOD = 0.5    # until the tempo is known
DEVICE_CHECK = 2.0      # how often to check if the default speaker changed
STEP_CHOICES = ((0.25, "¼"), (0.5, "½"), (1, "1"), (2, "2"), (3, "3"), (4, "4"))
RANDOM_STEPS = (1, 0.5, 0.25)
PACKET_RATE = 25        # lights acked ~29/s when i measured, more just queues up as lag
PACKET_BURST = 3
COLOUR_STEP = 16        # round colours so close ones share a packet

# lumisync defaults (lumisync/config/options.py)
LUMI_WINDOW = int(0.023 * 48000)
LUMI_GAIN = 1.7
LUMI_HIT = 0.55                   # their auto director calls this a hit
LUMI_HIT_GAP = 0.2

PRESETS = (
    (255, 0, 0), (255, 96, 0), (255, 200, 0), (0, 255, 0), (0, 255, 160),
    (0, 200, 255), (0, 0, 255), (140, 0, 255), (255, 0, 160), (255, 255, 255),
)

# colour 1, colour 2. bright and saturated only, dark shades just look off on leds
SCHEMES = {
    # holidays and classics
    "Police": ((255, 0, 0), (0, 0, 255)),
    "Christmas": ((255, 0, 0), (0, 255, 0)),
    "Halloween": ((255, 90, 0), (140, 0, 255)),
    "Valentine": ((255, 0, 60), (255, 90, 170)),
    "St Patrick": ((0, 255, 40), (255, 200, 0)),
    "Stars & stripes": ((255, 255, 255), (0, 60, 255)),
    # neon and party
    "Synthwave": ((255, 59, 127), (0, 245, 212)),
    "Cyberpunk": ((255, 0, 200), (0, 230, 255)),
    "Vaporwave": ((255, 113, 206), (1, 205, 254)),
    "Miami Vice": ((255, 45, 149), (0, 255, 170)),
    "Neon rave": ((57, 255, 20), (255, 110, 199)),
    "Electric": ((125, 249, 255), (188, 19, 254)),
    "Acid": ((180, 255, 0), (255, 0, 180)),
    "Toxic": ((160, 255, 0), (176, 0, 255)),
    "Laser": ((255, 0, 0), (0, 255, 255)),
    # nature
    "Sunset": ((255, 69, 0), (255, 0, 160)),
    "Sunrise": ((255, 180, 0), (255, 40, 80)),
    "Ocean": ((0, 255, 170), (0, 120, 255)),
    "Deep sea": ((0, 40, 255), (0, 255, 220)),
    "Aurora": ((0, 255, 140), (130, 0, 255)),
    "Forest": ((0, 255, 60), (160, 255, 0)),
    "Tropical": ((0, 255, 180), (255, 120, 0)),
    "Lava": ((255, 30, 0), (255, 140, 0)),
    "Fire & ice": ((255, 60, 0), (0, 200, 255)),
    "Ice": ((180, 240, 255), (0, 80, 255)),
    "Golden hour": ((255, 170, 0), (255, 90, 20)),
    "Cherry blossom": ((255, 110, 180), (255, 255, 255)),
    "Lavender": ((170, 90, 255), (255, 140, 220)),
    "Galaxy": ((90, 0, 255), (255, 0, 200)),
    "Northern sky": ((0, 120, 255), (0, 255, 120)),
    # pairs
    "Teal & coral": ((0, 200, 180), (255, 110, 70)),
    "Orange & blue": ((255, 140, 0), (0, 60, 255)),
    "Green & purple": ((0, 255, 60), (170, 0, 255)),
    "Royal": ((90, 0, 255), (255, 190, 0)),
    "Lemon lime": ((255, 240, 0), (120, 255, 0)),
    "Cotton candy": ((255, 140, 220), (120, 210, 255)),
    "Bubblegum": ((255, 60, 170), (160, 120, 255)),
    "Peacock": ((0, 210, 170), (40, 60, 255)),
    "Flamingo": ((255, 80, 140), (255, 150, 60)),
    "Watermelon": ((255, 30, 90), (40, 255, 80)),
    "Mojito": ((120, 255, 60), (255, 255, 255)),
    "Candlelight": ((255, 140, 30), (255, 80, 0)),
    "Warm & cool white": ((255, 180, 100), (200, 230, 255)),
}


# shared by the window, audio thread and lights thread
class Settings:
    def __init__(self):
        self.brightness = 0.8
        self.intensity = 0.7    # 0 = steady, 1 = follows the music fully
        self.sensitivity = 0.5
        self.detector = "aubio"  # or "lumisync"
        self.delay = 0.05       # send beats early to cover bluetooth lag
        self.scheme = "Police"    # or "Custom"
        self.color1, self.color2 = SCHEMES[self.scheme]
        self.random_colours = False
        self.steps_per_beat = 1
        self.offbeat = False      # ½ and ¼ step on the upbeat instead
        self.random_steps = False
        self.patterns = ["pulse"]
        self.switch_seconds = 20.0
        self.current: str | None = None  # set by the lights thread
        self.switch_at = 0.0


# low pass biquad (rbj cookbook)
def bass_filter(samplerate: int, cutoff: float = 150.0) -> aubio.digital_filter:
    w0 = 2 * np.pi * cutoff / samplerate
    alpha = np.sin(w0) / (2 * 0.707)
    cos = np.cos(w0)
    a0 = 1 + alpha
    f = aubio.digital_filter(3)
    f.set_biquad((1 - cos) / 2 / a0, (1 - cos) / a0, (1 - cos) / 2 / a0, -2 * cos / a0, (1 - alpha) / a0)
    return f


# listens to pc audio
# beats: aubio finds the bpm and locks to the beat grid -> evenly spaced
# lumi_hits: lumisync loudness jumps, no tempo, just reacts
class AudioAnalyser(threading.Thread):

    def __init__(self, settings: Settings, samplerate: int = 48000, size: int = 4096, hop: int = 512):
        super().__init__(daemon=True)
        self.settings = settings
        self.samplerate = samplerate
        self.size = size  # 4096 -> ~12 Hz bass resolution
        self.hop = hop    # 512 -> ~11 ms
        self.level = 0.0
        self.beats = 0
        self.bpm = 0.0
        self.last_beat = 0.0
        self.recent = np.zeros(LUMI_WINDOW)
        self.lumi_hits = 0
        self.lumi_level = 0.0
        self.last_hit = 0.0
        self.hit_period = DEFAULT_PERIOD  # smoothed gap between hits
        self.on_beat = lambda: None  # lights loop sets this to wake instantly
        self.device = "starting"
        self.error: Exception | None = None

    def run(self) -> None:
        warnings.filterwarnings("ignore", message=".*discontinuity.*")
        try:
            while True:
                try:
                    self.listen()
                except (RuntimeError, IndexError) as exc:  # device gone, e.g. bluetooth speaker turned off
                    self.device = f"reconnecting ({exc})"
                    self.level = 0.0
                    time.sleep(1)
        except Exception as exc:
            self.error = exc

    # records the default speaker until it changes (e.g. bluetooth speaker connects) then returns
    def listen(self) -> None:
        speaker = sc.default_speaker()
        loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        self.device = loopback.name
        next_check = time.monotonic() + DEVICE_CHECK

        tracker = aubio.tempo("default", 2 * self.hop, self.hop, self.samplerate)
        lowpass = bass_filter(self.samplerate)
        freqs = np.fft.rfftfreq(self.size, 1 / self.samplerate)
        bass_bins = (freqs >= 20) & (freqs <= 150)
        window = np.hanning(self.size)
        buffer = np.zeros(self.size)
        peak = 1.0
        lumi_gain = lumi.AutoGain(self.samplerate // self.hop)

        with loopback.recorder(samplerate=self.samplerate, channels=2, blocksize=self.hop) as rec:
            while True:
                if time.monotonic() >= next_check:
                    if sc.default_speaker().name != speaker.name:
                        return
                    next_check = time.monotonic() + DEVICE_CHECK

                new = rec.record(numframes=self.hop).mean(axis=1)
                buffer = np.concatenate((buffer[len(new):], new))
                self.recent = buffer[-LUMI_WINDOW:].copy()
                self.lumi_hit(lumi_gain)

                bass = np.abs(np.fft.rfft(buffer * window))[bass_bins].sum()
                peak = max(bass, peak * 0.995, 0.5)  # slowly forget loud moments
                self.level = min(bass / peak, 1.0)

                tracker.set_threshold(0.6 - 0.5 * self.settings.sensitivity)
                # bass only so it locks to the kick, not the hi-hats
                if tracker(lowpass(new.astype(np.float32)))[0]:
                    self.bpm = tracker.get_bpm()
                    self.last_beat = time.monotonic()
                    self.beats += 1
                    if self.settings.detector == "aubio":
                        self.on_beat()

    # same level/transient maths as their MusicPatternRenderer.render
    def lumi_hit(self, gain: lumi.AutoGain) -> None:
        bands = gain.process(lumi.spectral_bands(self.recent, self.samplerate))
        level = lumi.MusicPatternRenderer._level(sum(bands), LUMI_GAIN)
        transient = min(1.0, max(0.0, level - self.lumi_level) * 4.0)
        self.lumi_level = level
        threshold = LUMI_HIT + 0.3 * (0.5 - self.settings.sensitivity)  # sensitivity 50 -> their 0.55
        now = time.monotonic()
        if transient > threshold and now - self.last_hit > LUMI_HIT_GAP:
            gap = now - self.last_hit
            if gap < 2.0:
                self.hit_period = 0.8 * self.hit_period + 0.2 * gap
            self.last_hit = now
            self.lumi_hits += 1
            if self.settings.detector == "lumisync":
                self.on_beat()


# predicts beats from the tempo so they can go out early
class BeatClock:
    def __init__(self):
        self.count = 0
        self.last_target = 0.0  # predicted time of the last fired beat
        self.fired_at = 0.0

    def next_target(self, analyser: AudioAnalyser, now: float, delay: float) -> float | None:
        if analyser.bpm <= 0:
            return None
        period = 60 / analyser.bpm
        if now - analyser.last_beat > 2 * period:  # music stopped / lost the beat
            return None
        # next beat on the grid that isn't fired yet and isn't already too late
        n_unfired = math.floor((self.last_target + period / 2 - analyser.last_beat) / period) + 1
        n_on_time = math.ceil((now + delay - LATE_OK - analyser.last_beat) / period)
        n = max(0, n_unfired, n_on_time)
        return analyser.last_beat + n * period

    def fire(self, target: float, now: float) -> None:
        self.count += 1
        self.last_target = target
        self.fired_at = now


# ---------- patterns ----------
# Ctx -> 15 bulb colours. only changed bulbs get sent, 1 packet per colour, so keep colours few

@dataclass
class Ctx:
    beat: int           # steps so far
    since: float
    period: float       # seconds per step
    flash: float        # 1 on the beat, fades to the intensity floor
    loud: float         # follows bass loudness
    intensity: float
    c1: tuple
    c2: tuple
    rng: random.Random  # same numbers for the whole step

    # 0 -> 1 through the step
    @property
    def phase(self) -> float:
        return min(self.since / self.period, 1.0)

    def level(self, envelope: float) -> float:
        return quant(1 - self.intensity * 0.95 * (1 - envelope), 4)


def quant(x: float, steps: int) -> float:
    return round(x * steps) / steps


def dim(rgb: tuple, v: float) -> tuple:
    return tuple(int(c * v) for c in rgb)


def mix(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def hue(h: float) -> tuple:
    return tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h % 1.0, 1.0, 1.0))


# beats -> pattern steps. 2 splits every beat in half, ½ steps every 2nd beat
def to_steps(count: int, since: float, period: float, per_beat: float, offbeat: bool) -> tuple[int, float, float]:
    if per_beat >= 1:
        n = int(per_beat)
        length = period / n
        sub = min(n - 1, int(since / length))
        return count * n + sub, since - sub * length, length
    n = round(1 / per_beat)
    count += n // 2 if offbeat else 0  # halfway through the group
    return count // n, since + (count % n) * period, period * n


# --- any tempo ---

def p_single(c: Ctx) -> list:
    return [dim(c.c1, c.flash)] * BULBS


def p_rainbow(c: Ctx) -> list:
    return [dim(hue(c.beat * 0.17), c.flash)] * BULBS


def p_two(c: Ctx) -> list:
    return [dim(c.c2 if c.beat % 2 else c.c1, c.loud)] * BULBS


def p_alternate(c: Ctx) -> list:
    return [dim(c.c1 if (i + c.beat) % 2 == 0 else c.c2, c.flash) for i in range(BULBS)]


def p_fill(c: Ctx) -> list:
    # on one by one then off one by one
    step = c.beat % (2 * BULBS)
    lit = range(step) if step <= BULBS else range(step - BULBS, BULBS)
    return [dim(c.c1, c.flash) if i in lit else OFF for i in range(BULBS)]


def p_sparkle(c: Ctx) -> list:
    lit = c.rng.sample(range(BULBS), 6)
    return [dim(c.c1, c.flash) if i in lit else OFF for i in range(BULBS)]


# --- slow songs ---

def p_breathe(c: Ctx) -> list:
    # colour 1 -> 2 -> 1 over two steps, shifted early so it peaks on the beat
    t = ((c.beat % 2 + (c.since + BREATHE_LEAD) / c.period) / 2) % 1.0
    return [mix(c.c1, c.c2, quant((1 - math.cos(2 * math.pi * t)) / 2, 8))] * BULBS


def p_heartbeat(c: Ctx) -> list:
    # lub dub
    dub_at = min(0.25, 0.3 * c.period)
    lub = math.exp(-c.since / 0.1)
    dub = 0.7 * math.exp(-(c.since - dub_at) / 0.1) if c.since >= dub_at else 0.0
    return [dim(c.c1, c.level(max(lub, dub)))] * BULBS


def p_running(c: Ctx) -> list:
    # bands of 3 roll along one place per step
    return [dim(c.c1 if ((i + c.beat) // 3) % 2 == 0 else c.c2, c.flash) for i in range(BULBS)]


# --- fast songs ---

# string split into n parts that swap colour every step. 2 -> halves
def sections(n: int):
    def pattern(c: Ctx) -> list:
        return [dim(c.c1 if (i * n // BULBS + c.beat) % 2 == 0 else c.c2, c.flash) for i in range(BULBS)]
    return pattern


def p_theater(c: Ctx) -> list:
    # every third bulb, moves one per step
    return [dim(c.c1, c.flash) if (i + c.beat) % 3 == 0 else dim(c.c2, 0.15) for i in range(BULBS)]


PATTERNS = {  # key: (label, function)
    "pulse": ("Pulse", p_single),
    "rainbow": ("Rainbow", p_rainbow),
    "two": ("Two colours (swap on beat)", p_two),
    "alternate": ("Alternate bulbs", p_alternate),
    "fill": ("Fill on / off one by one", p_fill),
    "sparkle": ("Sparkle", p_sparkle),
    "breathe": ("Breathe (slow)", p_breathe),
    "heartbeat": ("Heartbeat (slow)", p_heartbeat),
    "running": ("Running bands (slow)", p_running),
    "halves": ("Halves (swap on beat)", sections(2)),
    "thirds": ("Thirds (swap on beat)", sections(3)),
    "quarters": ("Quarters (swap on beat)", sections(4)),
    "theater": ("Theater chase (fast)", p_theater),
}


# does the switch timer do anything
def switches(settings: Settings) -> bool:
    return len(settings.patterns) > 1 or settings.random_steps or settings.random_colours


def choose_pattern(settings: Settings, now: float, on_beat: bool) -> None:
    ticked = settings.patterns
    if not ticked:
        settings.current = None
    elif settings.current not in ticked:
        settings.current = ticked[0]
        settings.switch_at = now + settings.switch_seconds
    elif switches(settings) and now >= settings.switch_at and (on_beat or now >= settings.switch_at + 2):
        # on a beat, or after 2s if no beat comes
        settings.current = ticked[(ticked.index(settings.current) + 1) % len(ticked)]
        if settings.random_steps:
            settings.steps_per_beat = random.choice([v for v in RANDOM_STEPS if v != settings.steps_per_beat])
        if settings.random_colours:
            settings.scheme = random.choice([name for name in SCHEMES if name != settings.scheme])
            settings.color1, settings.color2 = SCHEMES[settings.scheme]
        settings.switch_at = now + settings.switch_seconds


# token bucket so we never send faster than the lights can take
class PacketBudget:

    def __init__(self):
        self.tokens = float(PACKET_BURST)
        self.updated = time.monotonic()

    def available(self) -> int:
        now = time.monotonic()
        self.tokens = min(PACKET_BURST, self.tokens + (now - self.updated) * PACKET_RATE)
        self.updated = now
        return int(self.tokens)

    def spend(self, n: int) -> None:
        self.tokens -= n


def snap(rgb: tuple) -> tuple:
    return tuple(min(255, round(c / COLOUR_STEP) * COLOUR_STEP) for c in rgb)


# changed bulbs grouped by colour, biggest change first, as many as the budget allows
# skipped ones catch up next frame. shown = what the lights display now
async def send_frame(client, frame: list[tuple], shown: list[tuple], budget: PacketBudget) -> None:
    groups: dict[tuple, list[int]] = {}
    for i, rgb in enumerate(frame):
        rgb = snap(rgb)
        if rgb != shown[i]:
            groups.setdefault(rgb, []).append(i)

    def change(item: tuple) -> int:
        rgb, bulbs = item
        return sum(abs(a - b) for i in bulbs for a, b in zip(rgb, shown[i]))

    chosen = sorted(groups.items(), key=change, reverse=True)[:budget.available()]
    budget.spend(len(chosen))
    for rgb, bulbs in chosen:
        await send(client, color(*rgb, sum(1 << i for i in bulbs)))
        for i in bulbs:
            shown[i] = rgb


async def drive(settings: Settings, analyser: AudioAnalyser, stop: threading.Event) -> None:
    beat_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def wake() -> None:
        # runs on the audio thread. a beat landing right as stop closes the loop gets dropped
        try:
            loop.call_soon_threadsafe(beat_event.set)
        except RuntimeError:
            pass

    analyser.on_beat = wake
    try:
        await run_lights(settings, analyser, stop, beat_event)
    finally:
        analyser.on_beat = lambda: None  # the audio thread outlives this loop


async def run_lights(settings: Settings, analyser: AudioAnalyser, stop: threading.Event,
                     beat_event: asyncio.Event) -> None:
    async with connect(ADDRESS) as client:
        await send(client, power(True))

        smooth = 0.0
        clock = BeatClock()
        shown = [(-255, -255, -255)] * BULBS  # unknown -> every bulb gets sent first
        budget = PacketBudget()
        last_keep_alive = time.monotonic()
        seen_hits = analyser.lumi_hits

        while not stop.is_set():
            if not client.is_connected:
                raise RuntimeError("Lights disconnected")
            if analyser.error:
                raise RuntimeError(f"Audio capture failed: {analyser.error}")

            now = time.monotonic()
            if settings.detector == "lumisync":
                # no tempo so nothing to predict, fire on each hit
                on_beat = analyser.lumi_hits != seen_hits
                seen_hits = analyser.lumi_hits
                if on_beat:
                    clock.fire(now, now)
                beat_period = analyser.hit_period
            else:
                target = clock.next_target(analyser, now, settings.delay)
                on_beat = target is not None and now >= target - settings.delay
                if on_beat:
                    clock.fire(target, now)
                beat_period = 60 / analyser.bpm if analyser.bpm > 0 else DEFAULT_PERIOD
            choose_pattern(settings, now, on_beat)

            smooth = max(analyser.level, smooth * 0.85)  # fast rise, slow fall
            step, since, period = to_steps(clock.count, now - clock.fired_at, beat_period,
                                           settings.steps_per_beat, settings.offbeat)
            ctx = Ctx(
                beat=step,
                since=since,
                period=period,
                flash=0.0, loud=0.0,
                intensity=settings.intensity,
                c1=settings.color1, c2=settings.color2,
                rng=random.Random(step),
            )
            ctx.flash = ctx.level(math.exp(-since / FLASH_DECAY))
            ctx.loud = ctx.level(smooth)

            frame = PATTERNS[settings.current][1](ctx) if settings.current else [OFF] * BULBS
            frame = [dim(rgb, settings.brightness) for rgb in frame]
            await send_frame(client, frame, shown, budget)

            if now - last_keep_alive > 2:
                await send(client, KEEP_ALIVE)
                last_keep_alive = now

            # wait 1/FPS, until the next predicted beat, or wake on a detected beat
            wait = 1 / FPS
            target = None if settings.detector == "lumisync" else clock.next_target(analyser, time.monotonic(), settings.delay)
            if target is not None:
                wait = max(0.0, min(wait, target - settings.delay - time.monotonic()))
            try:
                await asyncio.wait_for(beat_event.wait(), wait)
            except TimeoutError:
                pass
            beat_event.clear()


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = Settings()
        self.analyser = AudioAnalyser(self.settings)
        self.analyser.start()
        self.thread: threading.Thread | None = None
        self.stop = threading.Event()
        self.error: str | None = None
        self.seen_beats = 0

        root.title("Govee")
        self.button = tk.Button(root, text="Start", width=12, command=self.toggle)
        self.button.pack(padx=10, pady=(10, 0))

        self.slider("Brightness", "brightness")
        self.slider("Intensity", "intensity")
        self.slider("Beat sensitivity", "sensitivity")
        self.slider("Delay compensation (ms)", "delay", to=400, scale=1000)

        detector = tk.Frame(root)
        detector.pack(pady=(5, 0))
        tk.Label(detector, text="Beat detection:").pack(side="left")
        self.detector = tk.StringVar(value=self.settings.detector)
        for value, text in (("aubio", "Beat tracker (aubio)"), ("lumisync", "LumiSync hits")):
            tk.Radiobutton(detector, text=text, variable=self.detector, value=value,
                           command=lambda: setattr(self.settings, "detector", self.detector.get())).pack(side="left")

        steps = tk.Frame(root)
        steps.pack(pady=(5, 0))
        tk.Label(steps, text="Steps per beat:").pack(side="left")
        self.steps = tk.DoubleVar(value=self.settings.steps_per_beat)
        for value, text in STEP_CHOICES:
            tk.Radiobutton(steps, text=text, variable=self.steps, value=value,
                           command=lambda: setattr(self.settings, "steps_per_beat", self.steps.get())).pack(side="left")

        step_options = tk.Frame(root)
        step_options.pack()
        for attr, text in (("offbeat", "Offbeat (½, ¼)"), ("random_steps", "Randomise (1, ½, ¼)")):
            self.checkbox(step_options, text, attr)

        boxes = tk.Frame(root)
        boxes.pack(padx=10, pady=5)
        self.ticks = {}
        for n, (key, (label, _)) in enumerate(PATTERNS.items()):
            self.ticks[key] = tk.BooleanVar(value=key in self.settings.patterns)
            tk.Checkbutton(boxes, text=label, variable=self.ticks[key], command=self.update_patterns).grid(
                row=n % 7, column=n // 7, sticky="w", padx=5)

        switch = tk.Frame(root)
        switch.pack(pady=(5, 0))
        tk.Label(switch, text="Switch every (s):").pack(side="left")
        self.switch_text = tk.StringVar(value=f"{self.settings.switch_seconds:g}")
        self.switch_text.trace_add("write", lambda *_: self.update_switch())
        self.switch_entry = tk.Entry(switch, textvariable=self.switch_text, width=5)
        self.switch_entry.pack(side="left")

        colours = tk.Frame(root)
        colours.pack(pady=(5, 0))
        self.swatches = {}
        for attr, text in (("color1", "Colour 1"), ("color2", "Colour 2")):
            tk.Label(colours, text=text).pack(side="left")
            swatch = tk.Button(colours, width=4, bg=self.hex(getattr(self.settings, attr)),
                               command=lambda a=attr: self.pick(a))
            swatch.pack(side="left", padx=(2, 10))
            self.swatches[attr] = swatch

        presets = tk.Frame(root)
        presets.pack()
        for rgb in PRESETS:
            chip = tk.Label(presets, width=2, bg=self.hex(rgb), relief="raised")
            chip.bind("<Button-1>", lambda _, c=rgb: self.set_colour("color1", c))
            chip.bind("<Button-3>", lambda _, c=rgb: self.set_colour("color2", c))
            chip.pack(side="left", padx=1)
        tk.Label(root, text="left-click: colour 1   ·   right-click: colour 2", fg="grey40").pack()

        scheme_row = tk.Frame(root)
        scheme_row.pack(pady=(5, 0))
        self.scheme_label = tk.Label(scheme_row, width=24, anchor="w")
        self.scheme_label.pack(side="left")
        self.checkbox(scheme_row, "Randomise colours", "random_colours")

        schemes = tk.Frame(root)
        schemes.pack(padx=10)
        for n, (name, pair) in enumerate(SCHEMES.items()):
            chip = tk.Frame(schemes, relief="raised", borderwidth=1, cursor="hand2")
            chip.grid(row=n // 11, column=n % 11, padx=1, pady=1)
            for rgb in pair:
                half = tk.Label(chip, width=1, bg=self.hex(rgb))
                half.pack(side="left")
                half.bind("<Button-1>", lambda _, nm=name: self.set_scheme(nm))
                half.bind("<Enter>", lambda _, nm=name: self.scheme_label.config(text=f"Scheme: {nm}"))
                half.bind("<Leave>", lambda _: self.show_scheme())
        self.show_scheme()

        # debug: flips black/white on every detected beat
        self.beat_box = tk.Label(root, text="beats: 0", height=4, bg="black", fg="white")
        self.beat_box.pack(fill="x", padx=10)
        self.device_label = tk.Label(root, fg="grey40")
        self.device_label.pack()

        self.status = tk.Label(root, text="Stopped")
        self.status.pack(pady=(0, 10))
        self.poll()
        self.watch_beats()

    def slider(self, label: str, attr: str, to: int = 100, scale: int = 100) -> None:
        widget = tk.Scale(self.root, label=label, from_=0, to=to, orient="horizontal",
                          command=lambda v: setattr(self.settings, attr, int(v) / scale))
        widget.set(round(getattr(self.settings, attr) * scale))
        widget.pack(fill="x", padx=10)

    def checkbox(self, parent: tk.Widget, text: str, attr: str) -> None:
        var = tk.BooleanVar(value=getattr(self.settings, attr))
        tk.Checkbutton(parent, text=text, variable=var,
                       command=lambda: setattr(self.settings, attr, var.get())).pack(side="left")

    def update_patterns(self) -> None:
        self.settings.patterns = [key for key, var in self.ticks.items() if var.get()]

    def update_switch(self) -> None:
        try:
            seconds = float(self.switch_text.get())
        except ValueError:
            seconds = 0
        if seconds <= 0:
            self.switch_entry.config(bg="#f4b4b4")  # not a number, keep the old one
            return
        self.switch_entry.config(bg="white")
        self.settings.switch_seconds = seconds
        self.settings.switch_at = time.monotonic() + seconds

    @staticmethod
    def hex(rgb: tuple[int, int, int]) -> str:
        return "#%02x%02x%02x" % rgb

    def pick(self, attr: str) -> None:
        rgb, _ = colorchooser.askcolor(self.hex(getattr(self.settings, attr)))
        if rgb:
            self.set_colour(attr, tuple(int(c) for c in rgb))

    def set_colour(self, attr: str, rgb: tuple) -> None:
        setattr(self.settings, attr, rgb)
        self.settings.scheme = "Custom"
        self.show_colours()

    def set_scheme(self, name: str) -> None:
        self.settings.color1, self.settings.color2 = SCHEMES[name]
        self.settings.scheme = name
        self.show_colours()

    def show_colours(self) -> None:
        for attr, swatch in self.swatches.items():
            swatch.config(bg=self.hex(getattr(self.settings, attr)))
        self.show_scheme()

    def show_scheme(self) -> None:
        self.scheme_label.config(text=f"Scheme: {self.settings.scheme}")

    def toggle(self) -> None:
        if self.thread and self.thread.is_alive():
            self.stop.set()
            return
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def worker(self) -> None:
        try:
            asyncio.run(drive(self.settings, self.analyser, self.stop))
        except Exception as exc:
            self.error = str(exc)

    def poll(self) -> None:
        running = self.thread is not None and self.thread.is_alive()
        self.button.config(text="Stop" if running else "Start")
        self.status.config(text=self.error or (self.now_playing() if running else "Stopped"))
        self.device_label.config(text=f"listening to: {self.analyser.device}")
        if self.steps.get() != self.settings.steps_per_beat:  # randomise changed it
            self.steps.set(self.settings.steps_per_beat)
        if self.swatches["color1"].cget("bg") != self.hex(self.settings.color1):  # randomise changed them
            self.show_colours()
        self.root.after(100, self.poll)

    def now_playing(self) -> str:
        s = self.settings
        if s.current is None:
            return "Tick at least one pattern"
        text = f"Now: {PATTERNS[s.current][0]}"
        if switches(s):
            text += f"   ·   switch in {max(0, s.switch_at - time.monotonic()):.0f} s"
        return text

    def watch_beats(self) -> None:
        a = self.analyser
        lumisync = self.settings.detector == "lumisync"
        beats = a.lumi_hits if lumisync else a.beats
        if beats != self.seen_beats:
            self.seen_beats = beats
            odd = beats % 2 == 1
            tempo = f"~{60 / a.hit_period:.0f} hits/min" if lumisync else f"bpm: {a.bpm:.0f}"
            self.beat_box.config(text=f"beats: {beats}   {tempo}", bg="white" if odd else "black",
                                 fg="black" if odd else "white")
        self.root.after(10, self.watch_beats)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
