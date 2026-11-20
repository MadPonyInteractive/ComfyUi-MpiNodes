"""Join MiniMax H3 windows in LATENT space (Cubric Studio MPI-1608 Phase 3). Pure maths, no ComfyUI imports at
module scope, so tests/ can load it with bare `import h3_window_join`.

A windowed run (Cubric Studio's Video Edit) renders a long clip as windows: window k+1 starts with window k's last
`context_frames` frames held as a masked prefix (MpiH3MaskedPrefix), so its head IS window k's tail, byte for byte,
on both streams. The film is therefore window 1 + every later window minus its held head - joined here once, before
the one decode, instead of decoding every window and joining pixels (a re-encode per seam, a level per seam).
"""
try:  # loaded as part of the pack
    from .h3 import FPS_I, AUDIO_LATENT_FPS, frame_spans, latent_frames, plan_context, snap_context_frames
except ImportError:  # loaded alone by tests/
    from h3 import FPS_I, AUDIO_LATENT_FPS, frame_spans, latent_frames, plan_context, snap_context_frames

# The carried head is a straight copy, so it matches to the bit; anything above this is a different window.
HEAD_TOLERANCE = 1e-4


def lead_steps(total_steps: int, lead_frames: int) -> int:
    """Latent steps that hold exactly the first `lead_frames` frames; raises when they end mid-step."""
    if lead_frames <= 0:
        return 0
    for k, (_, end) in enumerate(frame_spans(total_steps)):
        if end == lead_frames:
            return k + 1
        if end > lead_frames:
            break
    raise ValueError(f"A {lead_frames}-frame lead-in does not end on a latent step (the grid is 1, 4, 4, 4, 4 frames).")


def join_windows(pieces, context_frames: int, lead_frames: int = 0):
    """`pieces`: [(video, audio), ...] oldest first, each a full window's AV latent as plain tensors
    (video [1, C, T, H, W], audio [1, C, 2, A]). Returns (video, audio, report).

    Every later piece must open with the earlier one's tail (`plan_context` picks the same steps the masked prefix
    held); a piece that does not is someone else's window, and that raises rather than joining two clips with a jump.
    `lead_frames` are dropped from the FIRST piece only (a lead-in the window rendered before the film starts);
    the audio drops the same time rounded down to a whole step.
    """
    import torch

    if not pieces:
        raise ValueError("Nothing to join.")
    wanted = snap_context_frames(context_frames)
    if wanted == 0:
        raise ValueError(f"context_frames {context_frames} is below the 39-frame minimum.")
    v0, a0 = pieces[0]
    ls = lead_steps(v0.shape[2], lead_frames)
    la = lead_frames * AUDIO_LATENT_FPS // FPS_I
    video, audio = [v0[:, :, ls:]], [a0[..., la:]]
    lines = [f"window 1: {v0.shape[2]} steps" + (f", lead-in {lead_frames} frames dropped" if ls else "")]
    for i in range(1, len(pieces)):
        prev_v, prev_a = pieces[i - 1]
        v, a = pieces[i]
        if v.shape[0] != 1 or v.shape[1] != prev_v.shape[1] or v.shape[3:] != prev_v.shape[3:]:
            raise ValueError(f"Window {i + 1} has another shape {tuple(v.shape)} than window {i} {tuple(prev_v.shape)}.")
        steps, frames, audio_steps = plan_context(prev_v.shape[2], wanted)
        if steps == 0 or steps >= v.shape[2] or audio_steps >= a.shape[-1]:
            raise ValueError(f"Window {i + 1} is too short to hold a {wanted}-frame head.")
        dv = (prev_v[:, :, -steps:].to(v) - v[:, :, :steps]).abs().max().item()
        da = (prev_a[..., -audio_steps:].to(a) - a[..., :audio_steps]).abs().max().item()
        if dv > HEAD_TOLERANCE or da > HEAD_TOLERANCE:
            raise ValueError(
                f"Window {i + 1} does not open with window {i}'s last {frames} frames (video differs by {dv:.4f}, "
                f"audio by {da:.4f}): it is not the next window of this run.")
        video.append(v[:, :, steps:])
        audio.append(a[..., audio_steps:])
        lines.append(f"window {i + 1}: {v.shape[2]} steps, its {frames}-frame head ({steps} steps, "
                     f"{audio_steps} audio steps) matched and dropped")
    jv = torch.cat([t.to(video[-1]) for t in video], dim=2)
    ja = torch.cat([t.to(audio[-1]) for t in audio], dim=-1)
    lines.append(f"film: {jv.shape[2]} steps = {latent_frames(jv.shape[2])} frames, audio {ja.shape[-1]} steps")
    return jv, ja, "\n".join(lines)
