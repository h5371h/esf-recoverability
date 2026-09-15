"""
Natus Neurovisor (.e) File Parser

Faithful Python reimplementation of FieldTrip:
  read_nervus_header.m  (Brogger & Wagenaar, 2016)
  read_nervus_data.m

Format: index-based container, little-endian, NOT a flat binary array.
Data:   int16 per channel, each channel has its own section(s) in the MainIndex.
Scale:  per-channel TSInfo.resolution (double), multiplied after int16 read.
"""

import bisect
import itertools
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants (match FieldTrip) ───────────────────────────────────────────────
TSLABELSIZE = 64    # uint16 chars for TS label  → 128 bytes
LABELSIZE   = 32    # uint16 chars for sensor name → 64 bytes
DAYSECS           = 86400.0
DATETIMEMINUSFACTOR = 2209161600

# ── Event-type GUIDs (FieldTrip read_nervus_header.m) ───────────────────────
# These identify the *type* of an event record. The record's pktGUID (the
# 16 bytes at the start of every packet in the Events section) is always
# EVENTGUID — these are the secondary "evtGUID" stored further inside.
HCEVENT_ANNOTATION    = '{A5A95612-A7F8-11CF-831A-0800091B5BDA}'
HCEVENT_SEIZURE       = '{A5A95646-A7F8-11CF-831A-0800091B5BDA}'
HCEVENT_FORMATCHANGE  = '{08784382-C765-11D3-90CE-00104B6F4F70}'
HCEVENT_PHOTIC        = '{6FF394DA-D1B8-46DA-B78F-866C67CF02AF}'
HCEVENT_POSTHYPERVENT = '{481DFC97-013C-4BC5-A203-871B0375A519}'

_EVENT_TYPE_NAMES: Dict[str, str] = {
    HCEVENT_ANNOTATION:    'Annotation',
    HCEVENT_SEIZURE:        'Seizure',
    HCEVENT_FORMATCHANGE:  'FormatChange',
    HCEVENT_PHOTIC:        'Photic',
    HCEVENT_POSTHYPERVENT: 'PostHyperventilation',
}

# EVENTGUID (B799F680-72A4-11D3-93D3-00500400C148) stored in Windows
# mixed-endian form on disk. Used as the per-record packet GUID.
_EVT_PKT_GUID = bytes([
    0x80, 0xF6, 0x99, 0xB7, 0xA4, 0x72, 0xD3, 0x11,
    0x93, 0xD3, 0x00, 0x50, 0x04, 0x00, 0xC1, 0x48,
])


# ── GUID → friendly name tables ───────────────────────────────────────────────

_STATIC_TAGS: Dict[str, str] = {
    'ExtraDataStaticPackets': 'ExtraDataStaticPackets',
    'SegmentStream':          'SegmentStream',
    'DataStream':             'DataStream',
    'ExtraDataTags':          'ExtraDataTags',
    'InfoChangeStream':       'InfoChangeStream',
    'InfoGuids':              'InfoGuids',
    'Events':                 'Events',
    '{A271CCCB-515D-4590-B6A1-DC170C8D6EE2}': 'TSGUID',
    '{8A19AA48-BEA0-40D5-B89F-667FC578D635}': 'DERIVATIONGUID',
    '{F824D60C-995E-4D94-9578-893C755ECB99}': 'FILTERGUID',
    '{02950361-35BB-4A22-9F0B-C78AAA5DB094}': 'DISPLAYGUID',
    '{8E9421-70F5-11D3-8F72-00105A9AFD56}':   'FILEINFOGUID',
    '{E4138BC0-7733-11D3-8685-0050044DAAB1}': 'SRINFOGUID',
    '{C728E565-E5A0-4419-93D2-F6CFC69F3B8F}': 'EVENTTYPEINFOGUID',
    '{D01B34A0-9DBD-11D3-93D3-00500400C148}': 'AUDIOINFOGUID',
    '{BF7C95EF-6C3B-4E70-9E11-779BFFF58EA7}': 'CHANNELGUID',
    '{2DEB82A1-D15F-4770-A4A4-CF03815F52DE}': 'INPUTGUID',
    '{5B036022-2EDC-465F-86EC-C0A4AB1A7A91}': 'INPUTSETTINGSGUID',
    '{99A636F2-51F7-4B9D-9569-C7D45058431A}': 'PHOTICGUID',
    '{55C5E044-5541-4594-9E35-5B3004EF7647}': 'ERRORGUID',
    '{223A3CA0-B5AC-43FB-B0A8-74CF8752BDBE}': 'VIDEOGUID',
    '{0623B545-38BE-4939-B9D0-55F5E241278D}': 'DETECTIONPARAMSGUID',
    '{CE06297D-D9D6-4E4B-8EAC-305EA1243EAB}': 'PAGEGUID',
    '{782B34E8-8E51-4BB9-9701-3227BB882A23}': 'ACCINFOGUID',
    '{3A6E8546-D144-4B55-A2C7-40DF579ED11E}': 'RECCTRLGUID',
    '{D046F2B0-5130-41B1-ABD7-38C12B32FAC3}': 'GUID TRENDINFOGUID',
    '{CBEBA8E6-1CDA-4509-B6C2-6AC2EA7DB8F8}': 'HWINFOGUID',
    '{E11C4CBA-0753-4655-A1E9-2B2309D1545B}': 'VIDEOSYNCGUID',
    '{B9344241-7AC1-42B5-BE9B-B7AFA16CBFA5}': 'SLEEPSCOREINFOGUID',
    '{15B41C32-0294-440E-ADFF-DD8B61C8B5AE}': 'FOURIERSETTINGSGUID',
    '{024FA81F-6A83-43C8-8C82-241A5501F0A1}': 'SPECTRUMGUID',
    '{8032E68A-EA3E-42E8-893E-6E93C59ED515}': 'SIGNALINFOGUID',
    '{30950D98-C39C-4352-AF3E-CB17D5B93DED}': 'SENSORINFOGUID',
    '{F5D39CD3-A340-4172-A1A3-78B2CDBCCB9F}': 'DERIVEDSIGNALINFOGUID',
    '{969FBB89-EE8E-4501-AD40-FB5A448BC4F9}': 'ARTIFACTINFOGUID',
    '{02948284-17EC-4538-A7FA-8E18BD65E167}': 'STUDYINFOGUID',
    '{D0B3FD0B-49D9-4BF0-8929-296DE5A55910}': 'PATIENTINFOGUID',
    '{7842FEF5-A686-459D-8196-769FC0AD99B3}': 'DOCUMENTINFOGUID',
    '{BCDAEE87-2496-4DF4-B07C-8B4E31E3C495}': 'USERSINFOGUID',
    '{B799F680-72A4-11D3-93D3-00500400C148}': 'EVENTGUID',
    '{AF2B3281-7FCE-11D2-B2DE-00104B6FC652}': 'SHORTSAMPLESGUID',
    '{89A091B3-972E-4DA2-9266-261B186302A9}': 'DELAYLINESAMPLESGUID',
    '{291E2381-B3B4-44D1-BB77-8CF5C24420D7}': 'GENERALSAMPLESGUID',
    '{5F11C628-FCCC-4FDD-B429-5EC94CB3AFEB}': 'FILTERSAMPLESGUID',
    '{728087F8-73E1-44D1-8882-C770976478A2}': 'DATEXDATAGUID',
    '{35F356D9-0F1C-4DFE-8286-D3DB3346FD75}': 'TESTINFOGUID',
}

# Raw hex (no dashes) → id_str for dynamic packet GUIDs
_DYNAMIC_GUIDS: Dict[str, str] = {
    'BF7C95EF6C3B4E709E11779BFFF58EA7': 'CHANNELGUID',
    '8A19AA48BEA040D5B89F667FC578D635': 'DERIVATIONGUID',
    'F824D60C995E4D949578893C755ECB99': 'FILTERGUID',
    '0295036135BB4A229F0BC78AAA5DB094': 'DISPLAYGUID',
    '782B34E88E514BB997013227BB882A23': 'ACCINFOGUID',
    'A271CCCB515D4590B6A1DC170C8D6EE2': 'TSGUID',
    'D01B34A09DBD11D393D300500400C148': 'AUDIOINFOGUID',
    '8E942170F511D38F7200105A9AFD56':   'FILEINFOGUID',
    'E4138BC0773311D386850050044DAAB1': 'SRINFOGUID',
    'C728E565E5A0441993D2F6CFC69F3B8F': 'EVENTTYPEINFOGUID',
    '2DEB82A1D15F4770A4A4CF03815F52DE': 'INPUTGUID',
    '5B0360222EDC465F86ECC0A4AB1A7A91': 'INPUTSETTINGSGUID',
    '99A636F251F74B9D9569C7D45058431A': 'PHOTICGUID',
    '55C5E044554145949E355B3004EF7647': 'ERRORGUID',
    '223A3CA0B5AC43FBB0A874CF8752BDBE': 'VIDEOGUID',
    '0623B54538BE4939B9D055F5E241278D': 'DETECTIONPARAMSGUID',
    'CE06297DD9D64E4B8EAC305EA1243EAB': 'PAGEGUID',
    '3A6E8546D1444B55A2C740DF579ED11E': 'RECCTRLGUID',
    'D046F2B0513041B1ABD738C12B32FAC3': 'GUID TRENDINFOGUID',
    'CBEBA8E61CDA4509B6C26AC2EA7DB8F8': 'HWINFOGUID',
    'E11C4CBA07534655A1E92B2309D1545B': 'VIDEOSYNCGUID',
    'B93442417AC142B5BE9BB7AFA16CBFA5': 'SLEEPSCOREINFOGUID',
    '15B41C320294440EADFFDD8B61C8B5AE': 'FOURIERSETTINGSGUID',
    '024FA81F6A8343C88C82241A5501F0A1': 'SPECTRUMGUID',
    '8032E68AEA3E42E8893E6E93C59ED515': 'SIGNALINFOGUID',
    '30950D98C39C4352AF3ECB17D5B93DED': 'SENSORINFOGUID',
    'F5D39CD3A3404172A1A378B2CDBCCB9F': 'DERIVEDSIGNALINFOGUID',
    '969FBB89EE8E4501AD40FB5A448BC4F9': 'ARTIFACTINFOGUID',
    '0294828417EC4538A7FA8E18BD65E167': 'STUDYINFOGUID',
    'D0B3FD0B49D94BF08929296DE5A55910': 'PATIENTINFOGUID',
    '7842FEF5A686459D8196769FC0AD99B3': 'DOCUMENTINFOGUID',
    'BCDAEE8724964DF4B07C8B4E31E3C495': 'USERSINFOGUID',
    'B799F68072A411D393D300500400C148': 'EVENTGUID',
    'AF2B32817FCE11D2B2DE00104B6FC652': 'SHORTSAMPLESGUID',
    '89A091B3972E4DA29266261B186302A9': 'DELAYLINESAMPLESGUID',
    '291E2381B3B444D1BB778CF5C24420D7': 'GENERALSAMPLESGUID',
    '5F11C628FCCC4FDDB4295EC94CB3AFEB': 'FILTERSAMPLESGUID',
    '728087F873E144D18882C770976478A2': 'DATEXDATAGUID',
    '35F356D90F1C4DFE8286D3DB3346FD75': 'TESTINFOGUID',
}


# ── Helper ────────────────────────────────────────────────────────────────────

def _decode_utf16le(data: bytes) -> str:
    return data.decode('utf-16-le', errors='ignore').rstrip('\x00').strip()


def _guid_reorder(mixed: List[int]) -> List[int]:
    """Apply standard Windows GUID byte-order reordering."""
    return [mixed[3], mixed[2], mixed[1], mixed[0],
            mixed[5], mixed[4], mixed[7], mixed[6],
            mixed[8], mixed[9], mixed[10], mixed[11],
            mixed[12], mixed[13], mixed[14], mixed[15]]


def _guid_to_formatted(reordered: List[int]) -> str:
    h = ''.join(f'{b:02X}' for b in reordered)
    return '{%s-%s-%s-%s-%s}' % (h[0:8], h[8:12], h[12:16], h[16:20], h[20:32])


def _guid_to_raw(reordered: List[int]) -> str:
    return ''.join(f'{b:02X}' for b in reordered)


# ── Main parser class ─────────────────────────────────────────────────────────

class NatusEParser:
    """
    Parse Natus Neurovisor .e files.

    Usage:
        parser = NatusEParser()
        result = parser.parse('/path/to/file.e')
        # result['signal'] shape: (n_samples, n_channels), float64
        # result['channels']: list of channel label strings
        # result['sfreq']: float
    """

    def parse(self, filepath: str) -> Dict[str, Any]:
        filepath = Path(filepath)
        logger.info(f"Parsing {filepath.name}  ({filepath.stat().st_size / 1e6:.1f} MB)")

        with open(filepath, 'rb') as f:
            hdr = self._read_header(f)
            static_packets = self._read_static_packets(f)
            qi = self._read_qi_index(f, len(static_packets))
            main_index = self._read_main_index(f, hdr['index_idx'], qi['nr_entries'])
            dynamic_packets = self._read_dynamic_packets(f, static_packets, main_index)
            ts_info = self._read_ts_info(f, static_packets, dynamic_packets, main_index)
            segments = self._read_segments(f, static_packets, main_index, ts_info)
            events = self._read_events(f, static_packets, main_index, segments)

        logger.info(f"  {len(ts_info)} channels in TSInfo")
        logger.info(f"  {len(segments)} segment(s)")
        for i, seg in enumerate(segments):
            srs = set(seg['sampling_rates'])
            logger.info(f"  Seg {i}: duration={seg['duration']:.2f}s  sfreqs={srs}")

        # Determine target channels: mode sampling rate (FieldTrip behaviour)
        sr_counts: Dict[float, int] = {}
        for sr in segments[0]['sampling_rates']:
            sr_counts[sr] = sr_counts.get(sr, 0) + 1
        target_sr = max(sr_counts, key=lambda k: sr_counts[k])
        matching_ch = [i for i, sr in enumerate(segments[0]['sampling_rates'])
                       if sr == target_sr]

        labels = [ts_info[i]['label'] for i in matching_ch]
        logger.info(f"  Target SR={target_sr} Hz, {len(matching_ch)} channels: {labels}")

        # Total samples across all segments for first matching channel
        first_ch = matching_ch[0]
        n_samples_total = sum(seg['sample_counts'][first_ch] for seg in segments)

        signal = self._read_signal(
            filepath, static_packets, main_index, segments,
            matching_ch, n_samples_total
        )

        return {
            'signal':       signal,           # (n_samples, n_channels) float64
            'channels':     labels,
            'sfreq':        target_sr,
            'n_channels':   len(matching_ch),
            'n_samples':    n_samples_total,
            'duration_sec': n_samples_total / target_sr,
            'all_ts_info':  ts_info,
            'events':       events,           # list[{onset_sec, duration_sec, label}]
        }

    # ── Header ────────────────────────────────────────────────────────────────

    def _read_header(self, f) -> Dict:
        f.seek(0)
        misc1 = struct.unpack('<5I', f.read(20))
        unknown = struct.unpack('<I', f.read(4))[0]
        index_idx = struct.unpack('<I', f.read(4))[0]
        if index_idx == 0:
            raise ValueError("Unsupported old-style Nicolet file (pre-~2012)")
        logger.debug(f"  indexIdx=0x{index_idx:08x}")
        return {'misc1': misc1, 'unknown': unknown, 'index_idx': index_idx}

    # ── Static packets (offset 172) ───────────────────────────────────────────

    def _read_static_packets(self, f) -> List[Dict]:
        f.seek(172)
        nr = struct.unpack('<I', f.read(4))[0]
        packets = []
        for _ in range(nr):
            tag_raw = struct.unpack('<40H', f.read(80))          # 40 uint16 = 80 bytes
            tag = ''.join(chr(c) for c in tag_raw if c != 0).strip()
            index = struct.unpack('<I', f.read(4))[0]
            id_str = _STATIC_TAGS.get(tag, tag if tag.isdigit() else 'UNKNOWN')
            packets.append({'tag': tag, 'index': index, 'id_str': id_str})
        logger.debug(f"  {nr} static packets")
        return packets

    # ── QI index (offset 172208) ──────────────────────────────────────────────

    def _read_qi_index(self, f, nr_static: int) -> Dict:
        f.seek(172208)
        nr_entries  = struct.unpack('<I', f.read(4))[0]
        misc1       = struct.unpack('<I', f.read(4))[0]
        qi_idx      = struct.unpack('<I', f.read(4))[0]
        misc3       = struct.unpack('<I', f.read(4))[0]
        lqi         = struct.unpack('<Q', f.read(8))[0]
        first_idx   = list(struct.unpack(f'<{nr_static}Q', f.read(8 * nr_static)))
        return {'nr_entries': nr_entries, 'lqi': lqi, 'first_idx': first_idx}

    # ── Main index (pointer chain) ────────────────────────────────────────────

    def _read_main_index(self, f, index_idx: int, nr_entries: int) -> List[Dict]:
        entries = []
        cur_count = 0
        next_ptr = index_idx

        while cur_count < nr_entries and next_ptr != 0:
            f.seek(next_ptr)
            nr_idx = struct.unpack('<Q', f.read(8))[0]
            raw = struct.unpack(f'<{3 * nr_idx}Q', f.read(24 * nr_idx))
            for i in range(nr_idx):
                section_idx = raw[3 * i]
                offset      = raw[3 * i + 1]
                combined    = raw[3 * i + 2]
                block_l     = combined & 0xFFFFFFFF
                section_l   = combined >> 32
                entries.append({
                    'section_idx': section_idx,
                    'offset':      offset,
                    'block_l':     block_l,
                    'section_l':   section_l,
                })
            next_ptr  = struct.unpack('<Q', f.read(8))[0]
            cur_count += nr_idx

        logger.debug(f"  MainIndex: {len(entries)} entries")
        return entries

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def _find_main(self, main_index: List[Dict], section_idx: int) -> Optional[Dict]:
        for m in main_index:
            if m['section_idx'] == section_idx:
                return m
        return None

    def _find_all_main(self, main_index: List[Dict], section_idx: int) -> List[int]:
        return [i for i, m in enumerate(main_index) if m['section_idx'] == section_idx]

    def _find_static(self, packets: List[Dict], id_str: str) -> Optional[Dict]:
        for p in packets:
            if p['id_str'] == id_str:
                return p
        return None

    # ── Dynamic packets (InfoChangeStream) ───────────────────────────────────

    def _read_dynamic_packets(self, f, static_packets: List[Dict],
                              main_index: List[Dict]) -> List[Dict]:
        ics = self._find_static(static_packets, 'InfoChangeStream')
        if ics is None:
            return []

        # FieldTrip uses MainIndex(indexIdx) directly; try sectionIdx lookup first,
        # fall back to direct array index (1-based) as FieldTrip does.
        main_ics = self._find_main(main_index, ics['index'])
        if main_ics is None:
            arr_idx = ics['index'] - 1  # convert 1-based to 0-based
            if 0 <= arr_idx < len(main_index):
                main_ics = main_index[arr_idx]
            else:
                logger.warning("Cannot locate InfoChangeStream section; no dynamic packets")
                return []

        nr_dyn = main_ics['section_l'] // 48
        f.seek(main_ics['offset'])
        packets = []

        for _ in range(nr_dyn):
            mixed = list(struct.unpack('16B', f.read(16)))
            reordered = _guid_reorder(mixed)
            guid_raw = _guid_to_raw(reordered)
            guid_fmt = _guid_to_formatted(reordered)
            id_str   = _DYNAMIC_GUIDS.get(guid_raw, 'UNKNOWN')

            date1    = struct.unpack('<d', f.read(8))[0]
            date2    = struct.unpack('<d', f.read(8))[0]  # FieldTrip reads two doubles for date
            datefrac = struct.unpack('<d', f.read(8))[0]
            internal_offset_start = struct.unpack('<Q', f.read(8))[0]
            packet_size           = struct.unpack('<Q', f.read(8))[0]

            packets.append({
                'guid':                   guid_raw,
                'guid_as_str':            guid_fmt,
                'id_str':                 id_str,
                'internal_offset_start':  internal_offset_start,
                'packet_size':            packet_size,
                'data':                   None,
            })

        # Load data only for TSGUID packets (what we actually need)
        for dp in packets:
            if dp['id_str'] == 'TSGUID' and dp['packet_size'] > 0:
                dp['data'] = self._load_dyn_data(f, dp, static_packets, main_index)

        return packets

    def _load_dyn_data(self, f, dp: Dict, static_packets: List[Dict],
                       main_index: List[Dict]) -> Optional[bytes]:
        sp = next((p for p in static_packets if p['tag'] == dp['guid_as_str']), None)
        if sp is None:
            return None
        instances_pos = self._find_all_main(main_index, sp['index'])
        if not instances_pos:
            return None

        buf = bytearray()
        internal_off = 0
        remaining    = dp['packet_size']
        target_start = dp['internal_offset_start']

        for pos in instances_pos:
            inst = main_index[pos]
            inst_end = internal_off + inst['section_l']
            if internal_off <= target_start < inst_end:
                start_at  = target_start
                stop_at   = min(start_at + remaining, inst_end)
                read_len  = stop_at - start_at
                f.seek(inst['offset'] + (start_at - internal_off))
                buf.extend(f.read(read_len))
                remaining    -= read_len
                target_start += read_len
            internal_off += inst['section_l']
            if remaining <= 0:
                break

        return bytes(buf)

    # ── TSInfo ────────────────────────────────────────────────────────────────

    def _read_ts_info(self, f, static_packets: List[Dict],
                      dynamic_packets: List[Dict],
                      main_index: List[Dict]) -> List[Dict]:
        # Prefer dynamic TSGUID packets (newer files)
        ts_dyn = [dp for dp in dynamic_packets
                  if dp['id_str'] == 'TSGUID' and dp['packet_size'] > 0
                  and dp['data'] is not None]
        if ts_dyn:
            logger.debug("  TSInfo source: dynamic packets")
            # dynamic path: elems at byte 752 (0-indexed), data starts at byte 760
            return self._parse_ts_info_blob(ts_dyn[0]['data'], elems_off=752, data_off=760)

        # Fall back to static TSGUID packet
        ts_sp = self._find_static(static_packets, 'TSGUID')
        if ts_sp is None:
            raise ValueError("No TSGUID packet found in file")

        main_entry = self._find_main(main_index, ts_sp['index'])
        if main_entry is None:
            raise ValueError(f"TSGUID section {ts_sp['index']} not in MainIndex")

        f.seek(main_entry['offset'])
        f.read(16)                                          # GUID bytes (skip)
        packet_len = struct.unpack('<Q', f.read(8))[0]     # uint64
        data = f.read(packet_len)
        logger.debug(f"  TSInfo source: static packet, packetLen={packet_len}")
        # static path: elems at byte 728 (0-indexed), data starts at byte 736
        return self._parse_ts_info_blob(data, elems_off=728, data_off=736)

    def _parse_ts_info_blob(self, data: bytes, elems_off: int, data_off: int) -> List[Dict]:
        """
        TSInfo entry layout (552 bytes each, 0-indexed within entry):
          [0..63]    label        (64 bytes = 32 utf-16-le chars, internalOffset += 128)
          [64..127]  padding
          [128..159] activeSensor (32 bytes = 16 utf-16-le chars, internalOffset += 64)
          [160..191] padding
          [192..199] refSensor    (8 bytes  = 4 utf-16-le chars,  internalOffset += 8)
          [200..255] padding      (internalOffset += 56)
          [256..263] lowcut       double
          [264..271] hiCut        double
          [272..279] samplingRate double
          [280..287] resolution   double  ← the int16 scale factor
          [288..289] specialMark  uint16
          [290..291] notch        uint16
          [292..299] eeg_offset   double
          [300..551] remainder    (unused)
        """
        elems = struct.unpack_from('<I', data, elems_off)[0]
        result = []
        for i in range(elems):
            base = data_off + i * 552
            label       = _decode_utf16le(data[base      : base + 64])
            active_snsr = _decode_utf16le(data[base + 128: base + 160])
            ref_snsr    = _decode_utf16le(data[base + 192: base + 200])
            low_cut,    = struct.unpack_from('<d', data, base + 256)
            hi_cut,     = struct.unpack_from('<d', data, base + 264)
            sfreq,      = struct.unpack_from('<d', data, base + 272)
            resolution, = struct.unpack_from('<d', data, base + 280)
            spec_mark,  = struct.unpack_from('<H', data, base + 288)
            notch,      = struct.unpack_from('<H', data, base + 290)
            eeg_off,    = struct.unpack_from('<d', data, base + 292)
            result.append({
                'label':         label,
                'active_sensor': active_snsr,
                'ref_sensor':    ref_snsr,
                'low_cut':       low_cut,
                'hi_cut':        hi_cut,
                'sampling_rate': sfreq,
                'resolution':    resolution,
                'special_mark':  spec_mark,
                'notch':         notch,
                'eeg_offset':    eeg_off,
            })
        return result

    # ── Segments ──────────────────────────────────────────────────────────────

    def _read_segments(self, f, static_packets: List[Dict],
                       main_index: List[Dict],
                       ts_info: List[Dict]) -> List[Dict]:
        sp = self._find_static(static_packets, 'SegmentStream')
        if sp is None:
            raise ValueError("No SegmentStream in file")
        main_entry = self._find_main(main_index, sp['index'])
        if main_entry is None:
            raise ValueError(f"SegmentStream section {sp['index']} not in MainIndex")

        nr_seg = main_entry['section_l'] // 152
        f.seek(main_entry['offset'])
        segments = []
        for _ in range(nr_seg):
            date_ole = struct.unpack('<d', f.read(8))[0]
            f.read(8)                                       # 8 bytes skip → offset 16
            duration = struct.unpack('<d', f.read(8))[0]   # offset 24
            f.read(128)                                     # 128 bytes skip → 152 total

            sampling_rates = [ts['sampling_rate'] for ts in ts_info]
            scales         = [ts['resolution']    for ts in ts_info]
            sample_counts  = [int(round(sr * duration)) for sr in sampling_rates]
            segments.append({
                'date_ole':      date_ole,
                'duration':      duration,
                'sampling_rates': sampling_rates,
                'scales':        scales,
                'sample_counts': sample_counts,
            })
        return segments

    # ── Events (annotations / photic / seizure / etc.) ───────────────────────

    def _read_events(self, f, static_packets: List[Dict],
                     main_index: List[Dict],
                     segments: List[Dict]) -> List[Dict]:
        """
        Parse the 'Events' section of a Natus .e file.

        Faithful port of FieldTrip's read_nervus_header.m event loop
        (Barborica, Dec 2015). Each event packet:

          offset  size  field
          ------  ----  -----
          0       16    pktGUID            (== EVENTGUID, marks record start)
          16      8     pktLen             (uint64 — bytes from record start
                                            to start of next record)
          24      8     eventID            (skipped)
          32      8     evtDate            (double, OLE date — days since 1899-12-30 UTC)
          40      8     evtDateFraction    (double — sub-day fraction in seconds)
          48      8     duration           (double — event duration in seconds)
          56      48    padding            (skipped)
          104     24    evtUser            (12 uint16 chars — creator app name)
          128     8     evtTextLen         (uint64 — uint16 chars in optional annotation text)
          136     16    evtGUID            (event-type GUID, classifies the event)
          152     16    padding            (skipped)
          168     64    evtLabel           (32 uint16 chars — short label)
          [for annotations only:]
          232     32    padding
          264     evtTextLen*2  annotation (extra free-text payload)

        Returns a list of {onset_sec, duration_sec, label} dicts. onset_sec
        is computed relative to segments[0] start (the recording origin)
        since downstream consumers think in recording-relative time.
        """
        sp = next((p for p in static_packets if p['tag'] == 'Events'), None)
        if sp is None:
            logger.debug("  No 'Events' static packet — file has no events")
            return []

        all_pos = self._find_all_main(main_index, sp['index'])
        if not all_pos:
            logger.debug("  Events section has no MainIndex entries")
            return []

        if not segments:
            # Need a recording origin to compute relative onsets
            logger.warning("  No segments — cannot compute event onsets, skipping events")
            return []

        recording_origin_ole = segments[0]['date_ole']

        events: List[Dict] = []
        # The Events section can span multiple MainIndex entries; concatenate logically.
        for pos in all_pos:
            entry = main_index[pos]
            section_offset = entry['offset']
            section_len    = entry['section_l']
            end = section_offset + section_len
            cur = section_offset

            while cur + 24 <= end:
                f.seek(cur)
                pkt_guid = f.read(16)
                if pkt_guid != _EVT_PKT_GUID:
                    # End of valid event records (rest of section is padding)
                    break
                pkt_len = struct.unpack('<Q', f.read(8))[0]
                if pkt_len < 232 or cur + pkt_len > end:
                    logger.debug(
                        f"  Event packet at {cur} has implausible pkt_len={pkt_len}; stop"
                    )
                    break

                try:
                    # Skip eventID (8 bytes)
                    f.seek(8, 1)
                    evt_date          = struct.unpack('<d', f.read(8))[0]
                    evt_date_fraction = struct.unpack('<d', f.read(8))[0]
                    duration_sec      = struct.unpack('<d', f.read(8))[0]
                    f.seek(48, 1)
                    _evt_user_bytes   = f.read(24)  # 12 uint16 — creator app, not exposed
                    evt_text_len      = struct.unpack('<Q', f.read(8))[0]
                    evt_guid_raw      = list(f.read(16))
                    evt_guid_fmt      = _guid_to_formatted(_guid_reorder(evt_guid_raw))
                    f.seek(16, 1)
                    label_bytes       = f.read(64)
                    label             = _decode_utf16le(label_bytes)

                    annotation_text: Optional[str] = None
                    if evt_guid_fmt == HCEVENT_ANNOTATION and 0 < evt_text_len < 4096:
                        f.seek(32, 1)
                        ann_bytes = f.read(evt_text_len * 2)
                        annotation_text = _decode_utf16le(ann_bytes)
                except struct.error as e:
                    logger.warning(f"  Event packet at {cur} truncated: {e}; stop")
                    break

                # OLE-date → relative onset.
                # Using the date-difference (in days) × DAYSECS is more numerically
                # stable than going through POSIX, because the absolute POSIX values
                # are ~3.5e9 while differences are O(1e3).
                onset_sec = (
                    (evt_date - recording_origin_ole) * DAYSECS
                    + evt_date_fraction
                )

                # Build a human-readable label:
                #   "<TypeName>: <label>"  if label present
                #   "<TypeName>"           otherwise
                # For annotations, prefer the annotation text.
                type_name = _EVENT_TYPE_NAMES.get(evt_guid_fmt, 'Unknown')
                if annotation_text:
                    display_label = annotation_text
                elif label:
                    display_label = f"{type_name}: {label}"
                else:
                    display_label = type_name

                events.append({
                    'onset_sec':    float(onset_sec),
                    'duration_sec': float(duration_sec) if duration_sec >= 0 else 0.0,
                    'label':        display_label,
                })

                cur += pkt_len
                if len(events) > 100_000:
                    logger.warning("  >100k events — capping to prevent runaway")
                    break

        logger.info(f"  {len(events)} event(s) extracted from 'Events' section")
        return events

    # ── Signal reading ────────────────────────────────────────────────────────

    def _read_signal(self, filepath: Path, static_packets: List[Dict],
                     main_index: List[Dict], segments: List[Dict],
                     ch_indices: List[int], n_samples_total: int) -> np.ndarray:
        """
        Read int16 data for each channel in ch_indices, across all segments,
        concatenate, scale by TSInfo.resolution → float64 output.
        Mirrors read_nervus_data.m logic exactly.
        """
        n_ch = len(ch_indices)
        out = np.zeros((n_samples_total, n_ch), dtype=np.float64)

        # Cumulative durations for segment boundary math
        cumsum_dur = list(itertools.accumulate(seg['duration'] for seg in segments))
        cumsum_dur = [0.0] + cumsum_dur   # prepend 0

        with open(filepath, 'rb') as f:
            for out_ch, ch in enumerate(ch_indices):
                self._read_channel(
                    f, main_index, static_packets, segments,
                    cumsum_dur, ch, out, out_ch
                )

        return out

    def _read_channel(self, f, main_index, static_packets, segments,
                      cumsum_dur, ch_idx, out, out_col):
        # ch_idx is 0-based; MATLAB stored channels as tag = str(ch_idx)
        sp = next((p for p in static_packets if p['tag'] == str(ch_idx)), None)
        if sp is None:
            logger.warning(f"  No static packet for channel {ch_idx}")
            return

        section_idx = sp['index']

        # All MainIndex entries for this channel
        all_pos = self._find_all_main(main_index, section_idx)
        if not all_pos:
            logger.warning(f"  No MainIndex sections for channel {ch_idx}")
            return

        # Section lengths in int16 samples (sectionL is bytes, divide by 2)
        sec_len   = [main_index[p]['section_l'] // 2 for p in all_pos]
        cumsum_sec = [0] + list(itertools.accumulate(sec_len))

        sample_offset = 0  # write position in out[:,out_col]

        for seg_idx, seg in enumerate(segments):
            cur_sf = seg['sampling_rates'][ch_idx]
            mult   = seg['scales'][ch_idx]
            n_samp = seg['sample_counts'][ch_idx]

            if n_samp == 0 or cur_sf == 0:
                continue

            # How many samples from previous segments precede this one for this channel
            skip_values = int(round(cumsum_dur[seg_idx] * cur_sf))

            # firstSectionForSegment: last section whose cumsum ≤ skip_values
            fsfg = bisect.bisect_right(cumsum_sec, skip_values) - 1
            fsfg = max(fsfg, 0)

            # Re-zero cumulative lengths to segment start
            offset_len = [cs - cumsum_sec[fsfg] for cs in cumsum_sec]

            range_start = 1          # 1-indexed, like MATLAB range = [1 n_samp]
            range_end   = n_samp

            # firstSection: last k where offset_len[k] < range_start
            first_sec = 0
            for k in range(len(offset_len)):
                if offset_len[k] < range_start:
                    first_sec = k

            # lastSection: (first k where offset_len[k] >= range_end) - 1
            last_sec = len(offset_len) - 2
            for k in range(len(offset_len)):
                if offset_len[k] >= range_end:
                    last_sec = k - 1
                    break

            if last_sec < first_sec:
                continue

            use_pos   = all_pos [first_sec: last_sec + 1]
            use_len   = sec_len [first_sec: last_sec + 1]

            n_to_read = range_end - range_start + 1
            seg_buf   = np.zeros(n_to_read, dtype=np.float64)
            cur = 0

            # ── First (possibly partial) section ──
            csec = main_index[use_pos[0]]
            f.seek(csec['offset'])
            first_off  = range_start - offset_len[first_sec]   # 1-indexed start within section
            last_off_v = min(range_end, use_len[0])             # end within section
            lsec       = last_off_v - first_off + 1
            f.seek((first_off - 1) * 2, 1)                     # skip to first sample
            raw = np.frombuffer(f.read(lsec * 2), dtype='<i2').astype(np.float64)
            seg_buf[:len(raw)] = raw * mult
            cur = len(raw)

            if len(use_pos) > 1:
                # ── Full middle sections ──
                for j in range(1, len(use_pos) - 1):
                    csec = main_index[use_pos[j]]
                    f.seek(csec['offset'])
                    raw = np.frombuffer(f.read(use_len[j] * 2), dtype='<i2').astype(np.float64)
                    seg_buf[cur: cur + len(raw)] = raw * mult
                    cur += len(raw)

                # ── Last (possibly partial) section ──
                csec = main_index[use_pos[-1]]
                f.seek(csec['offset'])
                want = n_to_read - cur
                raw = np.frombuffer(f.read(want * 2), dtype='<i2').astype(np.float64)
                seg_buf[cur: cur + len(raw)] = raw * mult

            out[sample_offset: sample_offset + n_to_read, out_col] = seg_buf
            sample_offset += n_to_read


# ── Convenience: export to EDF ────────────────────────────────────────────────

def to_edf(result: Dict[str, Any], out_path: str) -> None:
    """
    Write parser output to EDF using pyedflib (pip install pyedflib).
    Only writes the signal matrix; no annotations or events.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pip install pyedflib")

    signal  = result['signal']          # (n_samples, n_ch)
    labels  = result['channels']
    sfreq   = result['sfreq']
    n_ch    = result['n_channels']

    with pyedflib.EdfWriter(str(out_path), n_ch, file_type=pyedflib.FILETYPE_EDF) as edf:
        for i, lbl in enumerate(labels):
            ch_sig = signal[:, i]
            edf.setSignalHeader(i, {
                'label':            lbl[:16],
                'dimension':        'uV',
                'sample_rate':      int(sfreq),
                'physical_min':     float(ch_sig.min()),
                'physical_max':     float(ch_sig.max()),
                'digital_min':      -32768,
                'digital_max':       32767,
                'transducer':       '',
                'prefilter':        '',
            })
        edf.writeSamples([signal[:, i] for i in range(n_ch)])


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(message)s',
    )

    if len(sys.argv) < 2:
        print("Usage: python natus_e_parser.py <file.e> [output.edf]")
        sys.exit(1)

    parser = NatusEParser()
    result = parser.parse(sys.argv[1])

    sig = result['signal']
    print(f"\n✓ Parsed successfully")
    print(f"  Channels ({result['n_channels']}): {result['channels']}")
    print(f"  Sampling rate: {result['sfreq']} Hz")
    print(f"  Samples: {result['n_samples']}  Duration: {result['duration_sec']:.2f} s")
    print(f"  Signal shape: {sig.shape}")
    print(f"  Value range: [{sig.min():.4f}, {sig.max():.4f}]")

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
        print(f"\nWriting EDF to {out_path} ...")
        to_edf(result, out_path)
        print("  Done.")

    # If a matching .edf already exists with the same stem, compare
    e_path   = Path(sys.argv[1])
    edf_path = e_path.with_suffix('.edf')
    if not edf_path.exists():
        edf_path = e_path.parent / (e_path.stem.split('_')[0] + '_export.edf')
    if edf_path.exists():
        print(f"\nComparing with {edf_path.name} ...")
        try:
            import pyedflib
            with pyedflib.EdfReader(str(edf_path)) as edf:
                ref_n  = edf.signals_in_file
                ref_sr = edf.getSampleFrequencies()[0]
                ref_ch = [edf.signal_label(i).strip() for i in range(ref_n)]
                ref_sig = np.array([edf.readSignal(i) for i in range(ref_n)]).T

            print(f"  Reference: {ref_n} channels @ {ref_sr} Hz, {ref_sig.shape[0]} samples")
            n = min(sig.shape[0], ref_sig.shape[0])
            m = min(sig.shape[1], ref_sig.shape[1])
            corrs = [np.corrcoef(sig[:n, j], ref_sig[:n, j])[0, 1] for j in range(m)]
            print(f"  Per-channel Pearson r: {[f'{c:.4f}' for c in corrs[:8]]}...")
            print(f"  Mean r: {np.mean(corrs):.4f}")
        except Exception as e:
            print(f"  Comparison failed: {e}")
