"""Small EDF reader for the fixed-rate, numbered Tononi channels.

Reads only a selected interval and selected fixed-rate EEG channels. EDF+
annotation signals are not decoded or used as EEG; selected variable-rate
signals are rejected.
"""

from dataclasses import dataclass
from typing import BinaryIO

import numpy as np


@dataclass(frozen=True)
class Header:
    labels: tuple[str, ...]
    units: tuple[str, ...]
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    samples_per_record: tuple[int, ...]
    n_records: int
    record_seconds: float
    header_bytes: int
    patient: str

    @property
    def duration(self) -> float:
        return self.n_records * self.record_seconds


def _integer(value: bytes) -> int:
    return int(value.decode("ascii").strip())


def header(stream: BinaryIO) -> Header:
    first = stream.read(256)
    if len(first) != 256 or first[:8].strip() != b"0":
        raise ValueError("Not a standard EDF header")
    n = _integer(first[252:256])
    size = _integer(first[184:192])
    count = _integer(first[236:244])
    seconds = float(first[244:252].decode("ascii").strip())
    if n < 1 or size != 256 * (n + 1) or count <= 0 or seconds <= 0:
        raise ValueError("Invalid or unsupported EDF header")
    block = stream.read(size - 256)
    if len(block) != size - 256:
        raise ValueError("Truncated EDF header")
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    fields = []
    offset = 0
    for width in widths:
        fields.append(tuple(block[offset + i * width:offset + (i + 1) * width].decode("ascii").strip() for i in range(n)))
        offset += width * n
    if any("EDF+" in x for x in fields[9]):
        raise ValueError("EDF+ annotations require a different reader")
    return Header(fields[0], fields[2], np.array(fields[3], float), np.array(fields[4], float),
                  np.array(fields[5], float), np.array(fields[6], float), tuple(map(int, fields[8])),
                  count, seconds, size, first[8:88].decode("ascii", errors="replace").strip())


def interval(stream: BinaryIO, h: Header, indices: list[int], start: float, end: float) -> np.ndarray:
    if not 0 <= start < end <= h.duration or len(set(indices)) != len(indices):
        raise ValueError("Invalid interval or duplicate channels")
    if any(i < 0 or i >= len(h.labels) for i in indices):
        raise ValueError("Channel index out of range")
    rates = [h.samples_per_record[i] / h.record_seconds for i in indices]
    if len(set(rates)) != 1 or not float(rates[0]).is_integer():
        raise ValueError("Selected channels need an identical integer sample rate")
    rate = int(rates[0])
    first_sample, last_sample = round(start * rate), round(end * rate)
    if abs(first_sample / rate - start) > 1e-6 or abs(last_sample / rate - end) > 1e-6:
        raise ValueError("Interval not aligned to sample grid")
    by_channel = [[] for _ in indices]
    selected = {channel: j for j, channel in enumerate(indices)}
    bytes_per_record = 2 * sum(h.samples_per_record)
    first_record = int(start // h.record_seconds)
    last_record = int(np.ceil(end / h.record_seconds))
    stream.seek(h.header_bytes + first_record * bytes_per_record)
    for _ in range(first_record, last_record):
        record = stream.read(bytes_per_record)
        if len(record) != bytes_per_record:
            raise ValueError("Truncated EDF samples")
        offset = 0
        for channel, count in enumerate(h.samples_per_record):
            if channel in selected:
                values = np.frombuffer(record, dtype="<i2", count=count, offset=offset).astype(np.float32)
                scale = (h.physical_max[channel] - h.physical_min[channel]) / (h.digital_max[channel] - h.digital_min[channel])
                by_channel[selected[channel]].append((values - h.digital_min[channel]) * scale + h.physical_min[channel])
            offset += count * 2
    offset = first_sample - round(first_record * h.record_seconds * rate)
    n_samples = last_sample - first_sample
    result = np.stack([np.concatenate(parts)[offset:offset + n_samples] for parts in by_channel])
    if result.shape != (len(indices), n_samples) or not np.isfinite(result).all():
        raise ValueError("Unexpected EDF shape or nonfinite signal")
    return result


def indices_for_numbers(h: Header, numbers: list[int]) -> list[int]:
    """Match actual EDF labels, never presume the channel number equals its file position."""
    out = []
    for number in numbers:
        choices = [i for i, label in enumerate(h.labels) if label in (f"Chan {number}", str(number))]
        if len(choices) != 1 or h.units[choices[0]] != "uV":
            raise ValueError(f"Missing/duplicate/non-uV EEG channel {number}")
        out.append(choices[0])
    return out
