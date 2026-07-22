"""
steam_audio_renderer_v2.py
=======================

A narrow Python binding/wrapper for Steam Audio 4.8.1, designed for the
Living Brown Noise / Dream Instigator project.

Hard-coded default DLL location:
    D:\github\living-brown-noise-dream-instigator\phonon.dll

Architecture
------------
SteamAudioRenderer
    Owns one Steam Audio context and one shared HRTF.

SteamAudioSource
    Owns one persistent IPLBinauralEffect and a latest-value source position.

The application updates a source position by copying three floats. No Steam
Audio calls occur during the setter. When process_mono() is called for the next
audio block, the source reads the latest position, calculates direction and
distance, and renders that block through its persistent binaural effect.

The returned value is ordinary NumPy stereo PCM:
    shape: (frames, 2)
    dtype: float32

That same output can be sent to sounddevice or written during accelerated
offline export.

Current scope
-------------
- Fixed listener at origin.
- Listener looks down Steam Audio's forward axis: (0, 0, -1).
- Mono point sources rendered to binaural stereo.
- Continuously updateable source position.
- Bilinear HRTF interpolation.
- Optional simple artistic distance attenuation.
- Persistent Steam Audio DSP state across blocks.
- No device ownership; sounddevice remains separate.
- No reverb, reflections, occlusion, or scene simulation yet.

Dependency
----------
    pip install numpy
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np


# =============================================================================
# Public defaults
# =============================================================================

DEFAULT_PROJECT_DIRECTORY: Final[Path] = Path(
    r"D:\github\living-brown-noise-dream-instigator"
)
DEFAULT_DLL_PATH: Final[Path] = (
    DEFAULT_PROJECT_DIRECTORY / "phonon.dll"
)

DEFAULT_SAMPLE_RATE: Final[int] = 44_100
DEFAULT_FRAME_SIZE: Final[int] = 1_024

STEAMAUDIO_VERSION_MAJOR: Final[int] = 4
STEAMAUDIO_VERSION_MINOR: Final[int] = 8
STEAMAUDIO_VERSION_PATCH: Final[int] = 1
STEAMAUDIO_VERSION: Final[int] = (
    (STEAMAUDIO_VERSION_MAJOR << 16)
    | (STEAMAUDIO_VERSION_MINOR << 8)
    | STEAMAUDIO_VERSION_PATCH
)


# =============================================================================
# Steam Audio constants
# =============================================================================

IPL_STATUS_SUCCESS: Final[int] = 0
IPL_STATUS_FAILURE: Final[int] = 1
IPL_STATUS_OUTOFMEMORY: Final[int] = 2
IPL_STATUS_INITIALIZATION: Final[int] = 3

IPL_SIMDLEVEL_AVX2: Final[int] = 3
IPL_CONTEXTFLAGS_VALIDATION: Final[int] = 1 << 0

IPL_HRTFTYPE_DEFAULT: Final[int] = 0
IPL_HRTFNORMTYPE_NONE: Final[int] = 0

IPL_HRTFINTERPOLATION_NEAREST: Final[int] = 0
IPL_HRTFINTERPOLATION_BILINEAR: Final[int] = 1

STATUS_NAMES: Final[dict[int, str]] = {
    IPL_STATUS_SUCCESS: "IPL_STATUS_SUCCESS",
    IPL_STATUS_FAILURE: "IPL_STATUS_FAILURE",
    IPL_STATUS_OUTOFMEMORY: "IPL_STATUS_OUTOFMEMORY",
    IPL_STATUS_INITIALIZATION: "IPL_STATUS_INITIALIZATION",
}

LOG_LEVEL_NAMES: Final[dict[int, str]] = {
    0: "INFO",
    1: "WARNING",
    2: "ERROR",
    3: "DEBUG",
}


# =============================================================================
# ctypes declarations
# =============================================================================

CALLBACK_FACTORY = ctypes.WINFUNCTYPE

IPLLogFunction = CALLBACK_FACTORY(
    None,
    ctypes.c_int,
    ctypes.c_char_p,
)


class IPLContextSettings(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("logCallback", IPLLogFunction),
        ("allocateCallback", ctypes.c_void_p),
        ("freeCallback", ctypes.c_void_p),
        ("simdLevel", ctypes.c_int),
        ("flags", ctypes.c_int),
    ]


class IPLAudioSettings(ctypes.Structure):
    _fields_ = [
        ("samplingRate", ctypes.c_int32),
        ("frameSize", ctypes.c_int32),
    ]


class IPLHRTFSettings(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("sofaFileName", ctypes.c_char_p),
        ("sofaData", ctypes.POINTER(ctypes.c_uint8)),
        ("sofaDataSize", ctypes.c_int32),
        ("volume", ctypes.c_float),
        ("normType", ctypes.c_int),
    ]


class IPLVector3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


class IPLAudioBuffer(ctypes.Structure):
    _fields_ = [
        ("numChannels", ctypes.c_int32),
        ("numSamples", ctypes.c_int32),
        ("data", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
    ]


class IPLBinauralEffectSettings(ctypes.Structure):
    _fields_ = [
        ("hrtf", ctypes.c_void_p),
    ]


class IPLBinauralEffectParams(ctypes.Structure):
    _fields_ = [
        ("direction", IPLVector3),
        ("interpolation", ctypes.c_int),
        ("spatialBlend", ctypes.c_float),
        ("hrtf", ctypes.c_void_p),
        ("peakDelays", ctypes.POINTER(ctypes.c_float)),
    ]


# =============================================================================
# Public data types
# =============================================================================

@dataclass(frozen=True, slots=True)
class Vector3:
    """Application-space position. +X right, +Y up, -Z ahead."""

    x: float
    y: float
    z: float

    def distance(self) -> float:
        return math.sqrt(
            self.x * self.x
            + self.y * self.y
            + self.z * self.z
        )

    def normalized_direction(self) -> "Vector3":
        length = self.distance()
        if length <= 1e-9:
            return Vector3(0.0, 0.0, -1.0)

        inverse = 1.0 / length
        return Vector3(
            self.x * inverse,
            self.y * inverse,
            self.z * inverse,
        )


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    position: Vector3
    spatial_blend: float
    distance_attenuation_enabled: bool


# =============================================================================
# Errors
# =============================================================================

class SteamAudioError(RuntimeError):
    pass


def _check_status(status: int, operation: str) -> None:
    if status == IPL_STATUS_SUCCESS:
        return

    status_name = STATUS_NAMES.get(
        int(status),
        f"UNKNOWN_STATUS_{status}",
    )
    raise SteamAudioError(
        f"{operation} failed: {status_name} ({status})"
    )


# =============================================================================
# Steam Audio API binding
# =============================================================================

class _SteamAudioAPI:
    """Low-level function binding. Internal to this module."""

    def __init__(self, dll_path: Path) -> None:
        self.dll_path = Path(dll_path)

        if not self.dll_path.exists():
            raise FileNotFoundError(
                f"Steam Audio DLL not found: {self.dll_path}"
            )

        self._dll_directory_handle = None
        if hasattr(os, "add_dll_directory"):
            self._dll_directory_handle = os.add_dll_directory(
                str(self.dll_path.parent)
            )

        try:
            self.dll = ctypes.WinDLL(str(self.dll_path))
        except OSError as exc:
            raise SteamAudioError(
                "Unable to load phonon.dll or one of its dependencies: "
                f"{exc}"
            ) from exc

        self._bind()

    def _function(self, name: str):
        try:
            return getattr(self.dll, name)
        except AttributeError as exc:
            raise SteamAudioError(
                f"Steam Audio export is missing: {name}"
            ) from exc

    def _bind(self) -> None:
        self.context_create = self._function(
            "iplContextCreate"
        )
        self.context_create.argtypes = [
            ctypes.POINTER(IPLContextSettings),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.context_create.restype = ctypes.c_int

        self.context_release = self._function(
            "iplContextRelease"
        )
        self.context_release.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.context_release.restype = None

        self.hrtf_create = self._function(
            "iplHRTFCreate"
        )
        self.hrtf_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(IPLAudioSettings),
            ctypes.POINTER(IPLHRTFSettings),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.hrtf_create.restype = ctypes.c_int

        self.hrtf_release = self._function(
            "iplHRTFRelease"
        )
        self.hrtf_release.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.hrtf_release.restype = None

        self.binaural_effect_create = self._function(
            "iplBinauralEffectCreate"
        )
        self.binaural_effect_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(IPLAudioSettings),
            ctypes.POINTER(IPLBinauralEffectSettings),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.binaural_effect_create.restype = ctypes.c_int

        self.binaural_effect_release = self._function(
            "iplBinauralEffectRelease"
        )
        self.binaural_effect_release.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.binaural_effect_release.restype = None

        self.binaural_effect_apply = self._function(
            "iplBinauralEffectApply"
        )
        self.binaural_effect_apply.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(IPLBinauralEffectParams),
            ctypes.POINTER(IPLAudioBuffer),
            ctypes.POINTER(IPLAudioBuffer),
        ]
        self.binaural_effect_apply.restype = ctypes.c_int

        self.audio_buffer_allocate = self._function(
            "iplAudioBufferAllocate"
        )
        self.audio_buffer_allocate.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(IPLAudioBuffer),
        ]
        self.audio_buffer_allocate.restype = ctypes.c_int

        self.audio_buffer_free = self._function(
            "iplAudioBufferFree"
        )
        self.audio_buffer_free.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(IPLAudioBuffer),
        ]
        self.audio_buffer_free.restype = None

    def close(self) -> None:
        if self._dll_directory_handle is not None:
            self._dll_directory_handle.close()
            self._dll_directory_handle = None


# =============================================================================
# Renderer and source
# =============================================================================

class SteamAudioRenderer:
    """
    Owns the shared Steam Audio context and HRTF.

    Create one renderer for the whole application. Create one SteamAudioSource
    per independently positioned sound source.
    """

    def __init__(
        self,
        *,
        dll_path: Path = DEFAULT_DLL_PATH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_size: int = DEFAULT_FRAME_SIZE,
        validation_enabled: bool = False,
        log_messages: bool = True,
    ) -> None:
        self.dll_path = Path(dll_path)
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.validation_enabled = bool(validation_enabled)
        self.log_messages = bool(log_messages)

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive")

        self._closed = False
        self._sources: set[SteamAudioSource] = set()

        self._log_callback = self._make_log_callback()
        self._api = _SteamAudioAPI(self.dll_path)

        self._context = ctypes.c_void_p()
        self._hrtf = ctypes.c_void_p()

        self._audio_settings = IPLAudioSettings(
            samplingRate=self.sample_rate,
            frameSize=self.frame_size,
        )

        self._create_context()
        try:
            self._create_hrtf()
        except Exception:
            self.close()
            raise

    def _make_log_callback(self):
        enabled = self.log_messages

        @IPLLogFunction
        def callback(level: int, message: bytes | None) -> None:
            if not enabled:
                return

            decoded = (
                message.decode("utf-8", errors="replace")
                if message
                else "<no message>"
            )
            level_name = LOG_LEVEL_NAMES.get(
                level,
                f"LEVEL {level}",
            )
            print(f"[Steam Audio {level_name}] {decoded}")

        return callback

    def _create_context(self) -> None:
        settings = IPLContextSettings(
            version=STEAMAUDIO_VERSION,
            logCallback=self._log_callback,
            allocateCallback=None,
            freeCallback=None,
            simdLevel=IPL_SIMDLEVEL_AVX2,
            flags=(
                IPL_CONTEXTFLAGS_VALIDATION
                if self.validation_enabled
                else 0
            ),
        )

        status = int(
            self._api.context_create(
                ctypes.byref(settings),
                ctypes.byref(self._context),
            )
        )
        _check_status(status, "iplContextCreate")

        if not self._context.value:
            raise SteamAudioError(
                "iplContextCreate returned a NULL context"
            )

    def _create_hrtf(self) -> None:
        settings = IPLHRTFSettings(
            type=IPL_HRTFTYPE_DEFAULT,
            sofaFileName=None,
            sofaData=None,
            sofaDataSize=0,
            volume=1.0,
            normType=IPL_HRTFNORMTYPE_NONE,
        )

        status = int(
            self._api.hrtf_create(
                self._context,
                ctypes.byref(self._audio_settings),
                ctypes.byref(settings),
                ctypes.byref(self._hrtf),
            )
        )
        _check_status(status, "iplHRTFCreate")

        if not self._hrtf.value:
            raise SteamAudioError(
                "iplHRTFCreate returned a NULL HRTF"
            )

    @property
    def closed(self) -> bool:
        return self._closed

    def create_source(
        self,
        *,
        position: Vector3 = Vector3(0.0, 0.0, -2.0),
        spatial_blend: float = 1.0,
        distance_attenuation_enabled: bool = False,
    ) -> "SteamAudioSource":
        if self._closed:
            raise SteamAudioError("Renderer is closed")

        source = SteamAudioSource(
            renderer=self,
            position=position,
            spatial_blend=spatial_blend,
            distance_attenuation_enabled=(
                distance_attenuation_enabled
            ),
        )
        self._sources.add(source)
        return source

    def _forget_source(self, source: "SteamAudioSource") -> None:
        self._sources.discard(source)

    def close(self) -> None:
        if self._closed:
            return

        for source in tuple(self._sources):
            source.close()

        if self._hrtf.value:
            self._api.hrtf_release(
                ctypes.byref(self._hrtf)
            )

        if self._context.value:
            self._api.context_release(
                ctypes.byref(self._context)
            )

        self._api.close()
        self._closed = True

    def __enter__(self) -> "SteamAudioRenderer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SteamAudioSource:
    """
    One independently positioned persistent binaural source.

    Parameter setters only publish the latest values. Steam Audio is called
    exclusively from process_mono(), normally once per generated audio block.
    """

    def __init__(
        self,
        *,
        renderer: SteamAudioRenderer,
        position: Vector3,
        spatial_blend: float,
        distance_attenuation_enabled: bool,
    ) -> None:
        self._renderer = renderer
        self._api = renderer._api
        self._closed = False

        self._parameter_lock = threading.Lock()
        self._position = position
        self._spatial_blend = float(
            np.clip(spatial_blend, 0.0, 1.0)
        )
        self._distance_attenuation_enabled = bool(
            distance_attenuation_enabled
        )

        self._effect = ctypes.c_void_p()
        self._output_buffer = IPLAudioBuffer()

        self._input_storage = (
            ctypes.c_float * renderer.frame_size
        )()
        self._input_channel_pointers = (
            ctypes.POINTER(ctypes.c_float) * 1
        )(
            ctypes.cast(
                self._input_storage,
                ctypes.POINTER(ctypes.c_float),
            )
        )
        self._input_buffer = IPLAudioBuffer(
            numChannels=1,
            numSamples=renderer.frame_size,
            data=self._input_channel_pointers,
        )

        self._create_effect()
        try:
            self._allocate_output_buffer()
        except Exception:
            self.close()
            raise

    def _create_effect(self) -> None:
        settings = IPLBinauralEffectSettings(
            hrtf=self._renderer._hrtf
        )

        status = int(
            self._api.binaural_effect_create(
                self._renderer._context,
                ctypes.byref(self._renderer._audio_settings),
                ctypes.byref(settings),
                ctypes.byref(self._effect),
            )
        )
        _check_status(
            status,
            "iplBinauralEffectCreate",
        )

        if not self._effect.value:
            raise SteamAudioError(
                "iplBinauralEffectCreate returned a NULL effect"
            )

    def _allocate_output_buffer(self) -> None:
        status = int(
            self._api.audio_buffer_allocate(
                self._renderer._context,
                2,
                self._renderer.frame_size,
                ctypes.byref(self._output_buffer),
            )
        )
        _check_status(
            status,
            "iplAudioBufferAllocate",
        )

        if (
            self._output_buffer.numChannels != 2
            or self._output_buffer.numSamples
            != self._renderer.frame_size
            or not self._output_buffer.data
        ):
            raise SteamAudioError(
                "Steam Audio returned an invalid output buffer"
            )

    @property
    def closed(self) -> bool:
        return self._closed

    def set_position(
        self,
        x: float,
        y: float,
        z: float,
    ) -> None:
        """Publish a new position. Performs no Steam Audio work."""
        with self._parameter_lock:
            self._position = Vector3(
                float(x),
                float(y),
                float(z),
            )

    def set_position_vector(self, position: Vector3) -> None:
        self.set_position(
            position.x,
            position.y,
            position.z,
        )

    def set_spatial_blend(self, value: float) -> None:
        """0 = dry duplicated mono, 1 = fully binaural."""
        with self._parameter_lock:
            self._spatial_blend = float(
                np.clip(value, 0.0, 1.0)
            )

    def set_distance_attenuation_enabled(
        self,
        enabled: bool,
    ) -> None:
        with self._parameter_lock:
            self._distance_attenuation_enabled = bool(enabled)

    def snapshot(self) -> SourceSnapshot:
        with self._parameter_lock:
            return SourceSnapshot(
                position=self._position,
                spatial_blend=self._spatial_blend,
                distance_attenuation_enabled=(
                    self._distance_attenuation_enabled
                ),
            )

    @staticmethod
    def _distance_gain(distance: float) -> float:
        """
        Conservative artistic attenuation.

        Unity at two meters. Disabled by default because source-distance
        behavior will be tuned separately in the main application.
        """
        safe_distance = max(0.05, float(distance))
        gain_db = -6.0 * math.log2(safe_distance / 2.0)
        gain_db = float(np.clip(gain_db, -24.0, 4.0))
        return 10.0 ** (gain_db / 20.0)

    def _process_mono_exact(
        self,
        mono: np.ndarray,
    ) -> np.ndarray:
        """
        Render exactly one engine-sized mono block to stereo float32.

        mono shape:
            (renderer.frame_size,)

        return shape:
            (renderer.frame_size, 2)
        """
        if self._closed:
            raise SteamAudioError("Source is closed")
        if self._renderer.closed:
            raise SteamAudioError("Renderer is closed")

        samples = np.asarray(mono, dtype=np.float32)

        if samples.ndim != 1:
            raise ValueError(
                f"Expected mono 1D input, got shape {samples.shape}"
            )
        if len(samples) != self._renderer.frame_size:
            raise ValueError(
                "_process_mono_exact requires exactly "
                f"{self._renderer.frame_size} samples; received "
                f"{len(samples)}"
            )

        # Copy the most recent parameters exactly once at the start of the
        # audio-block operation.
        state = self.snapshot()
        direction = state.position.normalized_direction()

        # Copy input to stable ctypes storage owned by this source.
        ctypes.memmove(
            self._input_storage,
            samples.ctypes.data,
            samples.nbytes,
        )

        params = IPLBinauralEffectParams(
            direction=IPLVector3(
                direction.x,
                direction.y,
                direction.z,
            ),
            interpolation=IPL_HRTFINTERPOLATION_BILINEAR,
            spatialBlend=state.spatial_blend,
            hrtf=self._renderer._hrtf,
            peakDelays=None,
        )

        # The Steam Audio C API declares this function as returning an effect
        # state rather than an error status; the persistent effect itself owns
        # continuity between successive calls.
        self._api.binaural_effect_apply(
            self._effect,
            ctypes.byref(params),
            ctypes.byref(self._input_buffer),
            ctypes.byref(self._output_buffer),
        )

        left = np.ctypeslib.as_array(
            self._output_buffer.data[0],
            shape=(self._renderer.frame_size,),
        )
        right = np.ctypeslib.as_array(
            self._output_buffer.data[1],
            shape=(self._renderer.frame_size,),
        )

        # Copy away from Steam Audio's reusable output memory.
        output = np.column_stack((left, right)).astype(
            np.float32,
            copy=True,
        )

        if state.distance_attenuation_enabled:
            output *= self._distance_gain(
                state.position.distance()
            )

        return output

    def process_mono(
        self,
        mono: np.ndarray,
    ) -> np.ndarray:
        """
        Render an arbitrary-length mono buffer to stereo float32.

        Steam Audio retains DSP state across each internal fixed-size frame.
        A final partial frame is zero-padded and trimmed.
        """
        samples = np.asarray(mono, dtype=np.float32)

        if samples.ndim != 1:
            raise ValueError(
                f"Expected mono 1D input, got shape {samples.shape}"
            )

        if len(samples) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        frame_size = self._renderer.frame_size
        chunks: list[np.ndarray] = []

        for start in range(0, len(samples), frame_size):
            valid_count = min(
                frame_size,
                len(samples) - start,
            )

            block = np.zeros(
                frame_size,
                dtype=np.float32,
            )
            block[:valid_count] = samples[
                start:start + valid_count
            ]

            rendered = self._process_mono_exact(block)
            chunks.append(rendered[:valid_count])

        return np.concatenate(chunks, axis=0)

    def process_stereo_bed(
        self,
        stereo: np.ndarray,
        *,
        spatial_amount: float,
    ) -> np.ndarray:
        """
        Preserve an existing stereo bed while adding a binaural contribution.

        spatial_amount:
            0.0 returns the original stereo signal.
            1.0 returns the mono-derived Steam Audio rendering.
        """
        stereo32 = np.asarray(
            stereo,
            dtype=np.float32,
        )

        if stereo32.ndim != 2 or stereo32.shape[1] != 2:
            raise ValueError(
                "Expected stereo input with shape (frames, 2), got "
                f"{stereo32.shape}"
            )

        amount = float(
            np.clip(spatial_amount, 0.0, 1.0)
        )

        if amount <= 0.0 or len(stereo32) == 0:
            return stereo32.copy()

        mono = 0.5 * (
            stereo32[:, 0]
            + stereo32[:, 1]
        )
        spatial = self.process_mono(mono)

        return (
            stereo32
            + (spatial - stereo32) * amount
        ).astype(np.float32, copy=False)

    def close(self) -> None:
        if self._closed:
            return

        if (
            self._output_buffer.data
            and self._renderer._context.value
        ):
            self._api.audio_buffer_free(
                self._renderer._context,
                ctypes.byref(self._output_buffer),
            )

        if self._effect.value:
            self._api.binaural_effect_release(
                ctypes.byref(self._effect)
            )

        self._renderer._forget_source(self)
        self._closed = True

    def __enter__(self) -> "SteamAudioSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


# =============================================================================
# Minimal self-test
# =============================================================================

def _self_test() -> None:
    print("Steam Audio renderer v2 wrapper self-test")
    print(f"DLL: {DEFAULT_DLL_PATH}")

    with SteamAudioRenderer(
        validation_enabled=True,
        log_messages=True,
    ) as renderer:
        source = renderer.create_source(
            position=Vector3(0.0, 0.0, -2.0),
            spatial_blend=1.0,
        )

        rng = np.random.default_rng(20260722)
        mono = (
            rng.standard_normal(renderer.frame_size)
            * 0.05
        ).astype(np.float32)

        stereo = source.process_mono(mono)

        print(f"Input shape:  {mono.shape}")
        print(f"Output shape: {stereo.shape}")
        print(f"Output dtype: {stereo.dtype}")
        print(
            "Output finite: "
            f"{bool(np.all(np.isfinite(stereo)))}"
        )
        print(
            "Output peak:   "
            f"{float(np.max(np.abs(stereo))):.6f}"
        )

        source.set_position(-1.0, 0.0, -1.0)
        moved = source.process_mono(mono)

        difference = float(
            np.sqrt(
                np.mean(
                    (moved.astype(np.float64)
                     - stereo.astype(np.float64)) ** 2
                )
            )
        )
        print(
            "Position-change RMS difference: "
            f"{difference:.6f}"
        )

        if stereo.shape != (renderer.frame_size, 2):
            raise RuntimeError("Unexpected stereo output shape")
        if not np.all(np.isfinite(stereo)):
            raise RuntimeError("Non-finite output samples")
        if difference <= 1e-8:
            raise RuntimeError(
                "Changing position did not change rendered output"
            )

    print("PASS: wrapper initialized, rendered, moved, and cleaned up.")


if __name__ == "__main__":
    _self_test()
