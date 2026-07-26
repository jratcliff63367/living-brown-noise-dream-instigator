# Dream Instigator Project Handoff

## Purpose

This document is intended to be attached to the beginning of a new
ChatGPT conversation together with the two current Python scripts. The
goal is for work to continue seamlessly without having to reconstruct
the design history.

------------------------------------------------------------------------

# Project Summary

The project is **Dream Instigator**, a Python-based generative sleep
audio engine.

It is NOT simply a brown-noise generator.

The guiding philosophy is that the soundscape should behave like a
**living organism** with slow physiology rather than a collection of
looping audio tracks.

Everything should evolve continuously and almost imperceptibly over
hours.

Primary use case: - Listening while falling asleep. - Secondary use
case: - Meditation.

Eventually this may become an Android application, but the immediate
goal is to perfect the synthesis engine. Exported multi-hour WAV files
are currently an acceptable delivery format.

------------------------------------------------------------------------

# Current Architecture

Subsystems:

-   2D evolving correlated brown noise
-   Dual-source 3D brown-noise layer
-   Breath synthesis
-   Synthesized heartbeat
-   Metabolism (central controller)
-   Dream motif engine (under active development)
-   Steam Audio spatialization

Everything except the renderer should remain data driven.

------------------------------------------------------------------------

# Brown Noise

The brown-noise foundation is considered largely complete.

It includes:

-   evolving spectral character
-   evolving stereo correlation
-   evolving 3D spatial layer
-   metabolism-driven parameter evolution

The only remaining engineering issue is a subtle audio hitch believed to
occur when Steam Audio source positions change.

Evidence:

-   occurs with dream motifs disabled
-   reproduced by manually dragging 3D position sliders
-   almost certainly related to position updates rather than the motion
    generator itself

Likely fix:

Interpolate spatial motion within audio blocks instead of changing Steam
Audio positions only once per render block.

------------------------------------------------------------------------

# Breath

Breath is intentionally a major part of the organism.

Important clarification:

Strong breath is GOOD.

It simply should not remain urgent forever.

Breath should largely follow Metabolism.

If metabolism enters an active state for several minutes, breath may
remain stronger during that period.

Current issue:

Metabolism currently suppresses breath too aggressively.

Raise its minimum presence.

Do NOT eliminate stronger breathing.

------------------------------------------------------------------------

# Heartbeat

Heartbeat is now synthesized instead of derived from brown noise.

This was a successful design change.

Heartbeat should always remain audible.

It can occasionally become extremely prominent.

However:

It must NEVER remain extremely prominent for long periods.

Heartbeat needs its own state machine:

Background → Approach → Prominent → Retreat → Recovery → Background

Recovery temporarily overrides metabolism.

Heartbeat controls BOTH:

-   gain
-   distance

together.

------------------------------------------------------------------------

# Metabolism

This is now the central controller.

This was formerly called Weather.

Metabolism owns:

-   brown-noise evolution
-   breath
-   heartbeat requests
-   3D activity
-   overall texture

Current philosophy:

Metabolism influences every subsystem.

Each subsystem interprets those signals differently.

Heartbeat is NOT dictated directly by metabolism.

Breath mostly is.

------------------------------------------------------------------------

# Dream Motifs

Design philosophy:

Always exactly TWO dream worlds exist.

One:

Dominant.

One:

Distant.

Workflow:

-   scan motif folders at startup
-   choose motifs using shuffle bag
-   never repeat until every motif used
-   dominant slowly approaches
-   distant remains barely audible
-   crossfade positions
-   retired motif replaced after fully receding

Effects:

Triggered effects originate at motif position.

Then travel:

toward listener

past listener

above listener

around listener

Need manual tuning tools before automation refinement.

------------------------------------------------------------------------

# Immediate Next Work

Highest priority:

Manual 3D motif positioning mode.

Instead of automatic movement:

Expose direct controls.

Motif A:

-   X
-   Y
-   Distance
-   Gain

Motif B:

-   X
-   Y
-   Distance
-   Gain

Need to discover good spatial ranges experimentally.

Automation should come afterward.

------------------------------------------------------------------------

# User Interface

Current UI exists primarily for engineering.

Eventually:

Advanced mode only.

Normal UI should become extremely simple.

Likely controls:

Activity (Deep Sleep ←→ Meditation)

Start

Stop

Export

Everything else hidden behind DEBUG_UI.

------------------------------------------------------------------------

# Activity Philosophy

Ultimate architecture:

Intent

↓

Metabolism

↓

Subsystems

Eventually one user slider should influence 100+ internal parameters.

The user expresses intent.

The organism determines behavior.

------------------------------------------------------------------------

# Things That Worked Well

-   Metabolism
-   3D brown noise
-   Synthesized heartbeat
-   Resting tendency concept
-   Dual dream worlds architecture

------------------------------------------------------------------------

# Current Bugs

1.  

Minor Steam Audio spatial hitch.

2.  

Heartbeat can remain too strong too long.

3.  

Breath currently becomes too weak during quiet periods.

4.  

Dream motif startup currently preloads too much data.

Should become lazy-loaded.

------------------------------------------------------------------------

# Coding Philosophy

Avoid one-off fixes.

Prefer systems.

Everything should be data driven.

Whenever possible create abstractions instead of special cases.

------------------------------------------------------------------------

# Important Design Goal

The operative word throughout this project is:

ALIVE.

The sound should never feel repetitive.

Never random.

Never mechanical.

Instead it should feel like an organism whose physiology evolves
continuously over many hours.
