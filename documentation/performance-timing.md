# Performance timing

Stage 1, the scene relay and H3 use one projection of the vocal timeline. Measured
word intervals determine vocal activity; a segment's earlier RMS onset does not
start mouth movement. Explicit instrumental coverage and gaps between measured
words produce silent performance phases. Missing coverage, uncertain evidence and
unresolved lyric alignment remain diagnostic conflicts.

Performance phases are timing instructions inside a creative camera shot. They
do not create a camera cut per word or pause. Frame boundaries use the project's
video FPS; sub-frame intervals may disappear from frame relays, but their exact
semantic intervals and words remain available to H3. Words spanning
a scene or creative shot boundary keep their identity and clipped intervals on
both sides. Their original midpoint assigns the text once; the other portion
continues the audio without restarting the word.

Each vocal event retains its own speaker and offscreen status. Offscreen vocals
do not instruct a visible subject to move their mouth. The compiler removes an
unambiguous contradictory closed-mouth instruction for the bound singer while
preserving instructions for other silent actors. Unresolved performance conflicts
use `h3.performance.*` validation codes. Compiler revision 42 invalidates earlier
compiled prompt checkpoints.

CPU tests check consistent inputs through the final prompt, including 24, 25 and
30 fps. They do not establish perceptual phoneme accuracy of generated video.
