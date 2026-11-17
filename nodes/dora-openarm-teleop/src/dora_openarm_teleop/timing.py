"""Hand-pose timing the webxr node and the clutch must agree on (docs/teleop.md).

    MIN_POSE_TIMEOUT <= POSE_TIMEOUT <= clutch fresh_s (--stale-ms) < SMOOTHER_RESET_GAP
                                        clutch fresh_s < clutch resume_gap_s (or 0 = off)

POSE_TIMEOUT: webxr republishes the last frame's hand poses on every tick until
that frame is this old, then drops it and resets the hand smoothers. Above
fresh_s, a frame stall between the two disarms the clutch (its grips go stale)
but reaches it as no pose gap, so the clutch skips its settle rule while webxr
only suspended its smoother: the pose slides once frames resume and a quick
re-squeeze anchors on the slide. At or below fresh_s, every stall long enough
to disarm the clutch also resets the smoother, and the first pose back is the
hand itself. The value sits 10 ms below fresh_s, not on it: webxr times frame
arrivals and the clutch grip deliveries, a few ms of dora jitter apart. With
both at 0.1 s, 73 of 793 disarming stalls re-engaged on a 66-70 mm slide;
none of 827 at 0.09 (measured in openarm-quest-teleop).

SMOOTHER_RESET_GAP: a hand silent longer than this has its smoother reset and
the latched neck pivot re-latched. Kept above fresh_s so that a re-latch only
happens after the clutch has seen the gap and let go; otherwise an engaged
target would shift by the head's motion.

MIN_POSE_TIMEOUT: the Quest sends frames 13.9 ms apart (72 Hz); a timeout near
that drops merely late frames and resets the smoothers on each, so the pose
jumps to the raw hand without the 1 m/s rate limit. 0.03 s leaves two frames.

resume_gap_s: an engaged arm keeps its grip across a gap up to this long. The
clutch only suspends after fresh_s, so a window at or below fresh_s bridges no
gap at all.
"""

POSE_TIMEOUT_S = 0.09
SMOOTHER_RESET_GAP_S = 0.12
MIN_POSE_TIMEOUT_S = 0.03
PIVOT_MODE = "latched"
# Comparisons of seconds parsed from text.
EPS_S = 1e-9
