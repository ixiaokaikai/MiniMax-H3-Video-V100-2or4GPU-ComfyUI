"""Retry-safe RGB-only color correction against the actual accepted predecessor."""
from collections import deque
import json
import torch

from .long_video import LONG_VIDEO_SCHEMA
from .long_video_delivery import _resolve_inside, _sha256_file
from .long_video_color_match_advanced import process_long_video_color_match


def correct_dual_segment_color(frames, root, chain_id, segment_index, parent_candidate_id, enabled=True):
    if not enabled or segment_index == 0:
        return frames, {'status': 'disabled' if not enabled else 'first_segment_identity',
            'enabled': bool(enabled), 'audio_touched': False, 'latent_touched': False}
    import av
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    if manifest.get('chain_id') != chain_id:
        raise ValueError('Color Match predecessor belongs to another chain')
    previous = manifest['segments'][segment_index - 1]
    if previous['index'] != segment_index - 1 or previous['candidate_id'] != parent_candidate_id:
        raise ValueError('Color Match predecessor candidate identity mismatch')
    path = _resolve_inside(root, root/previous['video_path'])
    if _sha256_file(path) != previous['video_sha256']:
        raise ValueError('Color Match accepted predecessor checksum changed')
    tail = deque(maxlen=5)
    count = 0
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 2
        for frame in container.decode(video=0):
            tail.append(torch.from_numpy(frame.to_ndarray(format='rgb24')).float()/255)
            count += 1
    if count != previous['frame_count'] or not tail:
        raise ValueError('Color Match accepted predecessor frame count changed')
    if _sha256_file(path) != previous['video_sha256']:
        raise ValueError('Color Match predecessor changed during decode')
    context = {'schema': LONG_VIDEO_SCHEMA, 'empty': False, 'metadata': {
        'chain_id': chain_id, 'source_segment_index': segment_index-1,
        'target_segment_index': segment_index}}
    output, _, report = process_long_video_color_match(frames, context, chain_id, segment_index,
        _reference_frames=torch.stack(list(tail)), _persist_state=False)
    payload = json.loads(report)
    payload.update(predecessor_candidate_id=parent_candidate_id,
        predecessor_video_sha256=previous['video_sha256'], state_policy='derive_from_verified_accepted_video_no_sidecar')
    return output, payload
