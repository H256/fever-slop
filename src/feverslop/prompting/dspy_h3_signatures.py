from typing import Any

from feverslop.prompting.dspy_runtime import H3SignatureBundle


def build_h3_signature_bundle(dspy_module: Any | None = None) -> H3SignatureBundle:
    if dspy_module is None:
        import dspy as dspy_module

    from feverslop.prompting.dspy_h3_models import (
        BaseVideoPrompt,
        ImageAnalysis,
        PromptPlan,
        ResolvedPromptPlan,
        ResolvedReference,
        RetentionAnalysis,
    )

    class AnalyzeImage(dspy_module.Signature):
        """Analyze only observable information in a reference image for video generation."""

        image: dspy_module.Image = dspy_module.InputField()
        intended_role: str = dspy_module.InputField()
        user_hint: str = dspy_module.InputField()
        analysis: ImageAnalysis = dspy_module.OutputField()

    class BuildPromptPlan(dspy_module.Signature):
        """Create a strict, generation-ready video plan from the supplied user instruction,
        references, relay segments, and metadata.

        The plan is shared by all generation modes. Do not assume full-reference output
        syntax here. Plan the requested video faithfully; the downstream renderer will
        apply the mode-specific output guide.

        Preserve explicit input requirements instead of summarizing them away.

        Priority:
        1. explicit user and relay-segment directions,
        2. subject identity and appearance,
        3. clothing, accessories, props, and environment,
        4. requested action and performance direction,
        5. composition and camera direction,
        6. temporal and shot continuity,
        7. optional stylistic elaboration.

        Treat supplied references according to their declared roles. A reference role is
        a semantic constraint, not merely metadata.

        For subject references, preserve identity-critical visible attributes such as
        face, hair, body characteristics, clothing, footwear, accessories, and props
        when provided.

        For environment references, preserve defining spatial and visual characteristics.

        For composition, motion, camera, keyframe, first-frame, last-frame, storyboard,
        temporal-structure, edit-source, continuation, and audio references, preserve
        the concrete effect appropriate to that role.

        Do not invent, rename, renumber, merge, or silently omit reference labels.

        Directing instructions are executable constraints. Preserve concrete information
        such as:
        - who is visible,
        - subject position and orientation,
        - pose and action,
        - action order,
        - required props,
        - wardrobe state,
        - facial/performance state,
        - environment,
        - framing,
        - camera angle and movement,
        - movement direction and speed,
        - lighting,
        - timing and shot boundaries,
        - generic spatial relations and visibility scope.

        For every planned active subject, record whether and where it is visible in each
        relevant shot. Preserve explicit spatial relations literally, including stage versus
        audience, foreground versus background, left/center/right, front/behind, and the
        subject's relation to required props. When a subject's declared role identifies a
        tool, instrument, vehicle, or other role-defining prop used by the scene, bind that
        prop explicitly to the same subject and do not transfer it to another subject.
        Anonymous people in an environment reference
        must never absorb, replace, or spatially relocate a named subject.

        Unless the scene explicitly says an actor is absent or off-screen, every actor
        reference declared in the input is an active named subject. Include it in the plan
        and keep it visibly distinct from anonymous environment people. If the input gives no
        exact position, choose a coherent separate layer rather than placing the named actor
        inside the crowd or silently dropping the actor.

        A subject mentioned only as an audio source, voice, reaction, or implied stage
        presence does not count as visible coverage. For every actor action in character
        motion, create a visual subject entry and attach that actor to the corresponding shot
        with a concrete position, visible pose, and action.

        Do not replace specific directions with generic equivalents. For example,
        "slow truck right" must not become "cinematic camera movement", and a specified
        outfit must not become "casual clothing".

        If relay_segments are supplied, preserve their temporal ordering and attach each
        action, prop, state, and camera instruction to the corresponding planned shot.

        For each semantic action that should continue across technical render segments,
        add a continuation_intents entry with a stable action_id, requires_continuation=true,
        and the desired semantic duration in seconds. This is creative intent only: never
        choose backend frame counts, workflow limits, or technical segment boundaries.
        If the user requests a hard cut after a planned shot, preserve it with
        hard_cut_after=true and do not mark that boundary as continuation. When the input
        is ambiguous, default to a normal cut rather than inventing continuation intent.

        Across shots, preserve identity, wardrobe, props, environment, spatial state, and
        action continuity unless the input explicitly changes them.

        When strict_fidelity=true, explicit source information is authoritative. Creativity
        may fill unspecified details but must not overwrite specified ones.

        Before returning the plan, verify internally that no explicit action, outfit,
        prop, environment, camera instruction, composition constraint, or relevant
        reference role has been lost.

        For full-reference generation, author enough concrete creative detail across the
        planned shot descriptions for the compiled detailed_description to contain 350-500
        English words. A single-shot plan does not permit a shorter description. Cover
        composition, subject appearance and position, environment, lighting, action and
        state changes, camera movement, current sound, and each reference's active point.
        Also provide style_opening as one or two concrete English sentences establishing
        the target video's visual style and lighting. Do not put a shot label, timestamp,
        section header, or backend reference label in style_opening.
        Write description, visible_action, performance, camera_behavior,
        environmental_motion, and transition_intent as complementary concrete prose.
        The deterministic compiler concatenates every populated field, so do not duplicate
        the same sentence across fields and do not leave the requested detail only implicit.

        `requested_music_intent` is authoritative when it is non-empty. In particular,
        `requested_music_intent=none` means the soundtrack/scene audio is already
        supplied elsewhere and you MUST return `music_intent=none` with
        `non_diegetic_music` omitted or null. Do not reinterpret referenced song audio
        as non-diegetic music. For requested/enabled `generate` or `reference`, provide
        a concrete non-diegetic music description. Scene vocals, instruments, and
        referenced soundtrack audio belong in the detailed description and audio
        references, not in this field.
        """

        mode: str = dspy_module.InputField()
        user_prompt: str = dspy_module.InputField()
        duration_seconds: float | None = dspy_module.InputField()
        references: list[ResolvedReference] = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        strict_fidelity: bool = dspy_module.InputField()
        requested_music_intent: str = dspy_module.InputField()
        relay_segments: list[dict[str, Any]] = dspy_module.InputField(default=[])
        plan: PromptPlan = dspy_module.OutputField()

    class RenderBasePrompt(dspy_module.Signature):
        """Render a production-ready MiniMax base-mode prompt from the resolved plan.

        The supplied base-mode guide is authoritative for output syntax and mode-specific
        formatting. The resolved plan is authoritative for the requested visual content,
        actions, continuity, composition, and camera direction.

        Do not summarize away concrete plan constraints.

        Preserve explicitly specified subject appearance, clothing, accessories, props,
        environment, pose, action sequence, composition, framing, camera angle, camera
        motion, lighting, and timing.

        For T2VA, follow the guide's text-to-video structure directly. For I2VA, FL2VA,
        and L2VA, preserve the exact first-frame, first-and-last-frame, or last-frame
        relationship defined by the plan and guide. Do not treat frame anchors as generic
        visual inspiration.

        When a reference image determines an initial or final state, preserve identity,
        clothing, colors, key objects, spatial relationships, pose, and composition as
        required by that mode. Describe the visible transition path rather than replacing
        it with a static summary.

        Do not replace specific instructions with generic cinematic prose. Preserve
        distinctions such as pan vs. truck, tilt vs. pedestal, zoom vs. push/dolly, and
        static vs. moving camera whenever the plan specifies them.

        Across shots or temporal phases, preserve wardrobe, props, environment, identity,
        and physical state unless an explicit change is requested.

        When strict_fidelity=true, the plan's concrete requirements are hard constraints.
        Creativity may elaborate unspecified details but must not override specified ones.

        Before returning the result, internally verify that every explicit action, wardrobe
        detail, prop, environment constraint, composition requirement, frame-anchor
        relationship, and camera instruction from the plan is represented in the generated
        prompt.
        """

        guide: str = dspy_module.InputField()
        mode: str = dspy_module.InputField()
        user_prompt: str = dspy_module.InputField()
        plan: ResolvedPromptPlan = dspy_module.InputField()
        references: list[ResolvedReference] = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        strict_fidelity: bool = dspy_module.InputField()
        music_intent: str = dspy_module.InputField()
        relay_segments: list[dict[str, Any]] = dspy_module.InputField(default=[])
        result: BaseVideoPrompt = dspy_module.OutputField()

    class RenderReferencePrompt(dspy_module.Signature):
        """Render a valid MiniMax H3 full-reference prompt from the supplied resolved plan.

        The Full-Reference guide is authoritative for output syntax, label semantics,
        section structure, dialogue formatting, retention markers, and audio handling.
        The resolved plan is authoritative for what must happen in the generated video.

        Your task is to compile the plan into explicit, generation-ready H3 prose without
        losing constraints. Do not reinterpret or simplify concrete requirements merely
        to make the prose shorter, smoother, or more cinematic.

        Hard rule: no silent information loss. Never silently omit or generalize subject
        identity, face, hairstyle, clothing, footwear, accessories, props, environment
        identity, spatial layout, pose, action, action order, interactions, performance
        state, composition, framing, camera angle, camera motion, movement direction,
        timing, lighting, continuity state, or reference role.

        Use this precedence:
        1. explicit user instruction and relay-segment direction,
        2. resolved plan,
        3. supplied reference descriptions and roles,
        4. Full-Reference guide,
        5. optional cinematic elaboration.

        Every defined <Subject N> that appears in the target must be concretely anchored
        in detailed_description. At its first appearance, state the exact label, relevant
        identity and appearance characteristics, current clothing and important accessories
        when defined, its position or composition in frame, and its current pose/action/state.
        Do not merely write the label and assume the reference will supply all details.

        Every actor reference declared in the input is an active named subject unless the
        scene explicitly marks it absent or off-screen. It must appear in subject_definitions
        and in detailed_description at every shot where its action is planned. A crowd or
        environment reference supplies anonymous extras only; it must never absorb or replace
        the named actor. If the plan does not state an exact position, keep the actor visibly
        separate from the crowd in a coherent foreground, midground, or stage layer rather
        than inventing that the actor stands among the audience.

        When a referenced actor represents one physical individual, preserve exactly one
        persistent visible instance unless the resolved plan explicitly requires multiple
        instances, cloning, mirroring, reflection, or another deliberate multi-instance
        effect. Without such an explicit plan, an actor must not appear in two positions at
        once, such as one instance at the front of the frame and another at that actor's
        instrument. Keep each role-defining instrument or prop attached to that actor's
        single planned instance.

        An audio mention, lyric source, crowd reaction, or phrase such as "stage presence"
        does not count as showing the actor. The actor label must occur in the visual shot
        prose itself with a concrete position and visible action. Do not omit the actor from
        later shots merely because the camera is moving through the environment.

        Spatial visibility is a hard generation requirement, not background metadata. If a
        subject is part of the planned action, describe that subject visibly in the shot where
        the action occurs. State its exact frame position and spatial relation to other subjects
        and the environment: foreground/midground/background, left/center/right, on stage,
        behind the crowd, in front of the crowd, or another relation explicitly supplied by
        the plan. Never place a subject inside an audience, off-screen, or in the wrong spatial
        layer merely because the environment reference contains similar people. Preserve every
        required prop in the same shot and bind it to the correct subject. If a subject is
        required across multiple planned shots, mention it in each such shot; do not rely on
        retention_analysis to imply visual presence.

        Role-defining props are identity-and-action bindings: a subject who is identified as
        operating a tool, instrument, vehicle, or other prop must remain visibly attached to
        that prop in the relevant shot unless the plan explicitly changes the binding. Do
        not let another referenced subject inherit it merely because both appear together.

        Resolve crowded compositions explicitly. An environment or crowd reference supplies
        setting and anonymous extras; it does not replace, duplicate, or absorb a named subject.
        When the plan distinguishes a performer from an audience, keep the performer visibly
        separate from the audience and preserve the stated stage/front/back relationship.

        Treat clothing, footwear, jewelry, accessories, makeup, and wearable props as
        persistent subject state. Establish concrete wardrobe at the first clear appearance
        and preserve it across later shots unless an explicit wardrobe change occurs. Do not
        replace concrete wardrobe with vague phrases such as "same outfit", "reference
        outfit", or "casual clothing".

        Referenced environments are persistent anchors. Instantiate identity-critical
        architecture, room geometry, materials, furniture, distinctive objects, landscape
        features, lighting sources, and spatial layout where defined. Do not reduce a
        concrete referenced environment to a generic location.

        Translate directing instructions into literal visible shot behavior. For each shot,
        preserve all relevant planned information: framing, camera angle, visible subjects,
        subject position and orientation, wardrobe/appearance state, initial pose/state,
        exact action progression, required props and interactions, facial/performance state,
        environment, camera movement including direction/speed/amplitude when specified,
        lighting/atmosphere, and the ending state relevant to continuity.

        Preserve left/right, toward/away, up/down, entering/exiting direction, hand choice,
        posture, and prop interaction whenever specified. Do not collapse action sequences
        into a plot summary.

        When relay_segments are present, treat them as binding temporal direction. Represent
        each segment in the corresponding shot or temporal portion. Do not skip a relay
        instruction, move its action into another segment, replace it with a merely similar
        action, omit a required prop, change instrumental/vocal state, or invent a conflicting
        action. If a relay segment contains a preserved source instruction, action, or prop
        requirement, keep it attached to that segment.

        Preserve explicit camera semantics. Distinguish pan from truck, tilt from pedestal or
        crane, zoom from physical push/dolly, arc/orbit from pan, static from moving camera,
        and rack focus from camera motion. Preserve direction, speed, amplitude, tracking, and
        framing behavior when specified. Do not reduce a specific camera instruction to
        generic prose such as "cinematic camera movement".

        Apply every reference according to its role: subject preserves visible identity and
        attributes; environment preserves location identity and layout; style transfers only
        stylistic characteristics; composition controls framing/spatial arrangement; frame
        roles provide concrete shot anchors; motion controls referenced movement/action;
        camera controls camera behavior; temporal_structure controls ordering/cuts/pacing;
        edit_source and continuation preserve the source-video relationship; audio roles
        preserve their exact copy/reference relationship.

        A reference is not successfully used merely because its label appears. Its assigned
        characteristics must affect the target description. When several constraints apply to
        one shot, combine them in that shot rather than leaving important anchors only in
        summary or retention_analysis.

        Emit retention entries as required by the guide and resolved plan. Retention claims
        must agree with what is actually written in detailed_description; do not claim full
        preservation when defining attributes were replaced or omitted.

        Before returning, audit continuity across shots. Unless explicitly changed, preserve
        person identity, face, hair, clothing, accessories, props, environment, spatial
        relationships, physical/action state, and lighting/time state. A cut does not reset
        these attributes.

        Finally, verify every planned shot and supplied reference: all involved subjects are
        present, required actions occur in order, props are present and used correctly,
        clothing and environment constraints are retained, composition/framing and camera
        instructions are retained, timing/relay instructions are retained, and relevant
        reference labels are actually applied. A label-only mention does not pass this audit.
        Repair missing coverage before returning. Do not expose the internal audit.
        """

        guide: str = dspy_module.InputField()
        user_prompt: str = dspy_module.InputField()
        plan: ResolvedPromptPlan = dspy_module.InputField()
        references: list[ResolvedReference] = dspy_module.InputField()
        notes: str = dspy_module.InputField()
        strict_fidelity: bool = dspy_module.InputField()
        music_intent: str = dspy_module.InputField()
        relay_segments: list[dict[str, Any]] = dspy_module.InputField(default=[])
        summary: str = dspy_module.OutputField()
        retention_analysis: list[RetentionAnalysis] = dspy_module.OutputField()
        detailed_description: str = dspy_module.OutputField()
        overall_soundscape: str = dspy_module.OutputField()
        non_diegetic_music: str | None = dspy_module.OutputField(
            desc=(
                "Audience-only background music prose, or null/N/A. "
                "Never include <Audio N> labels or audio-reference definitions here; "
                "put those in detailed_description and retention_analysis."
            ),
        )

    class JudgeFinalPrompt(dspy_module.Signature):
        """Judge the final prompt against the supplied plan and guide.

        Use only the supplied inputs. Do not reject unfamiliar identifiers or
        invent a semantic whitelist. Return observational feedback; do not
        rewrite the prompt. Return a JSON object with `verdict` exactly equal to
        `good` or `bad`; `pass` and `fail` are accepted and normalized by the
        application. Include an `issues` list (empty when there are none).
        """

        guide: str = dspy_module.InputField()
        final_prompt: str = dspy_module.InputField()
        authoritative_plan: str = dspy_module.InputField()
        references: list[ResolvedReference] = dspy_module.InputField()
        # Keep this as a raw object because the model may report section-level
        # feedback in field_issues; the application normalizes that response
        # into PromptJudgeResult after DSPy returns it.
        judge: dict[str, Any] = dspy_module.OutputField()

    return H3SignatureBundle(
        AnalyzeImage,
        BuildPromptPlan,
        RenderBasePrompt,
        RenderReferencePrompt,
        JudgeFinalPrompt,
    )


def build_dspy_signatures():
    bundle = build_h3_signature_bundle()
    return (
        bundle.analyze_image,
        bundle.build_prompt_plan,
        bundle.render_base_prompt,
        bundle.render_reference_prompt,
    )
