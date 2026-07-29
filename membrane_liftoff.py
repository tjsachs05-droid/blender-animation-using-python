"""
membrane_liftoff.py
===================

Blender Python (bpy) script that animates the dissolution of a sacrificial
layer during membrane liftoff.  Written for / tested against Blender 4.x-5.x.

Scene layout (bottom -> top along +Z):
    * Substrate      (thickest stack of octahedra)
    * Sacrificial    (middle stack -- this is the layer that erodes away)
    * Top film       (the "membrane" -- stays in place, does not fall)

Each layer is a lateral grid (GRID_X x GRID_Y) of stacks, and every octahedron
is now its OWN object.  That lets the individual octahedra in the sacrificial
layer separate and dissipate one-by-one for a more realistic, crumbling look.

Erosion proceeds strictly ring-by-ring from the OUTER EDGE of the square inward.
Each octahedron is tagged with a ring index (its distance to the nearest grid
edge), and each ring gets its own non-overlapping time window: the whole
outermost square shell dissipates before the next shell begins, so you clearly
see the intact region shrinking.  Corners and edge faces share a ring, so they
leave together.  Each octahedron flies off in a randomised, heavily agitated
direction, tumbling as it shrinks away.

HOW TO RUN
----------
1. Open Blender, go to the Scripting workspace, load this file, "Run Script".
   (Or:  blender --python membrane_liftoff.py )
2. Press play to preview.  The script switches 3D viewports to Material shading
   so the colors show up immediately.

SAVING A VIDEO -- see the "HOW TO SAVE THE VIDEO" section at the bottom of this
file for step-by-step instructions (including how to keep the transparent
background).

Everything tweakable lives in the CONFIG block below.

NOTE ON "STACKS": the 3 / 4 / 5 numbers are how many octahedra TALL each layer
is (substrate thickest, film thinnest), over a shared GRID_X x GRID_Y footprint.
Edit LAYER_HEIGHTS / GRID_X / GRID_Y if you meant something else.
"""

import bpy
import bmesh
import math
import random
from mathutils import Vector


# =============================================================================
# CONFIG  --  edit anything in this block
# =============================================================================

class CFG:
    # ---- Reproducibility -----------------------------------------------------
    RANDOM_SEED = 7                 # change for a different erosion pattern

    # ---- Scene housekeeping --------------------------------------------------
    CLEAR_SCENE = True              # wipe the current .blend contents first

    # ---- Octahedron geometry -------------------------------------------------
    OCTA_RADIUS = 0.5               # center-to-vertex distance of one octahedron
    OCTA_GAP_XY = 0.05              # extra horizontal gap between stacks
    OCTA_GAP_Z = 0.05               # extra vertical gap within a stack

    # ---- Lateral extent of every layer (shared grid of stacks) ---------------
    GRID_X = 6                      # stacks (columns) along X
    GRID_Y = 6                      # stacks (columns) along Y

    # ---- Layer thicknesses, in octahedra tall (the "stacks" numbers) ---------
    LAYER_HEIGHTS = {               # bottom -> top
        "substrate":   5,          # thickest
        "sacrificial": 4,          # the layer that erodes
        "film":        3,          # the membrane on top
    }
    LAYER_SEPARATION_Z = 0.15       # gap between adjacent layers

    # ---- Water molecules -----------------------------------------------------
    WATER_COUNT = 40                # how many water "molecules" to spawn
    WATER_RADIUS = 2.0 * OCTA_RADIUS  # ~ two octahedra across
    WATER_UV_SEGMENTS = 16          # sphere resolution (keep modest for speed)
    WATER_SPAWN_RING_MARGIN = 3.0   # how far out from the stack water starts
    WATER_SPAWN_Z_SPREAD = 2.0      # vertical spread of the spawn shell
    COLOR_WATER = (0.20, 0.70, 0.95, 1.0)   # cyan, semi-transparent
    WATER_ALPHA = 0.45
    # Water timeline (frames)
    WATER_FADE_IN_START = 1
    WATER_FADE_IN_END   = 20
    WATER_APPROACH_END  = 60        # water has reached the sacrificial layer
    WATER_DISSIPATE_END = 175       # water fully faded out

    # ---- Colors (RGBA) -- vivid, emissive ------------------------------------
    COLOR_SUBSTRATE   = (0.45, 0.05, 0.85, 1.0)   # vivid violet
    COLOR_SACRIFICIAL = (1.00, 0.30, 0.02, 1.0)   # vivid orange-red
    COLOR_FILM        = (0.00, 0.80, 1.00, 1.0)   # vivid cyan
    EMISSION_STRENGTH = 0.6         # self-lit glow so colors pop on render
    MATERIAL_ROUGHNESS = 0.35       # a little shine

    # ---- Erosion motion (individual octahedra) -------------------------------
    # Erosion proceeds strictly ring-by-ring from the OUTSIDE in: the whole
    # outermost square shell dissipates before the next shell starts, so the
    # intact region is clearly seen shrinking. Keep
    #   RING_INTERVAL >= RING_STAGGER + PIECE_DURATION
    # to guarantee no inner piece erodes while an outer ring is still intact.
    EROSION_START           = 30    # frame the outermost ring begins to loosen
    EROSION_RING_INTERVAL   = 34    # frames between one ring starting and the next
    EROSION_RING_STAGGER    = 8     # spread of start times WITHIN a single ring
    EROSION_PIECE_DURATION  = 22    # frames from loosen -> fully dissipated

    EROSION_STREAM_DISTANCE = 7.0   # how far a piece flies before vanishing
    EROSION_RADIAL_STRENGTH = 1.0   # outward (away-from-center) bias
    EROSION_AGITATION       = 1.8   # random-direction magnitude (large = chaotic)
    EROSION_LIFT_BIAS       = 0.3   # small +Z bias so pieces stream, not sink
    EROSION_SPIN            = 9.0   # radians of tumble as a piece flies off

    # ---- Rendering / viewport smoothness ------------------------------------
    TRANSPARENT_BG = True           # render with a transparent (alpha) background
    USE_EEVEE = True                # EEVEE is far faster than Cycles for playback
    EEVEE_SAMPLES = 16              # low samples = smooth realtime playback
    FPS = 24
    FRAME_START = 1
    FRAME_END = 185                 # long enough for the water to fully dissipate
    RESOLUTION_X = 1280
    RESOLUTION_Y = 720
    RESOLUTION_PERCENT = 100
    USE_SIMPLIFY = True
    SIMPLIFY_VIEWPORT_SUBDIV = 1
    SET_MATERIAL_SHADING = True     # flip 3D viewports to Material shading


# =============================================================================
# Scene housekeeping
# =============================================================================

def clear_scene():
    """Remove all objects and datablocks WITHOUT relying on operator context
    (bpy.ops clearing is fragile from the Scripting workspace and can leave
    stragglers that cause '.001' name clashes on the next run)."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for block_coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves,
                       bpy.data.lights, bpy.data.cameras):
        for block in list(block_coll):
            block_coll.remove(block)


def ensure_collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


# =============================================================================
# Geometry builders
# =============================================================================

# Octahedron faces, referencing the 6 verts in the order created below.
_OCTA_FACES = [
    (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),   # top 4
    (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),   # bottom 4
]


def make_octahedron_mesh(name, radius):
    """One octahedron (6 verts, 8 flat faces), centered on its own origin."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    v = [
        bm.verts.new((0, 0,  radius)),
        bm.verts.new((0, 0, -radius)),
        bm.verts.new(( radius, 0, 0)),
        bm.verts.new((-radius, 0, 0)),
        bm.verts.new((0,  radius, 0)),
        bm.verts.new((0, -radius, 0)),
    ]
    for a, b, c in _OCTA_FACES:
        bm.faces.new((v[a], v[b], v[c]))
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = False
    mesh.update()
    return mesh


def make_material(name, rgba, emission_strength=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = CFG.MATERIAL_ROUGHNESS
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if emission_strength > 0.0:
            for key in ("Emission Color", "Emission"):   # renamed in 4.0
                if key in bsdf.inputs:
                    bsdf.inputs[key].default_value = rgba
                    break
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength
    mat.diffuse_color = rgba   # viewport display color (Solid shading)
    if alpha < 1.0:
        for attr, val in (("blend_method", "BLEND"),
                          ("show_transparent_back", False)):
            if hasattr(mat, attr):
                setattr(mat, attr, val)
    return mat


def make_uv_sphere_mesh(name, radius, segments):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    seg = max(6, segments)
    rings = max(4, segments // 2)
    # create_uvsphere uses `radius` on Blender 3.0+; fall back to `diameter`.
    try:
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings,
                                  radius=radius)
    except TypeError:
        bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings,
                                  diameter=radius * 2.0)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


# =============================================================================
# Build the stacked layers (one object per octahedron)
# =============================================================================

def build_layers(materials):
    step_xy = 2.0 * CFG.OCTA_RADIUS + CFG.OCTA_GAP_XY
    step_z = 2.0 * CFG.OCTA_RADIUS + CFG.OCTA_GAP_Z

    x0 = -0.5 * (CFG.GRID_X - 1) * step_xy
    y0 = -0.5 * (CFG.GRID_Y - 1) * step_xy

    layer_order = ["substrate", "sacrificial", "film"]
    objects = {name: [] for name in layer_order}
    bounds = {}

    z_cursor = 0.0
    for name in layer_order:
        height = CFG.LAYER_HEIGHTS[name]
        coll = ensure_collection(f"Layer_{name}")

        # One single-octahedron mesh per layer -> one material -> correct color,
        # shared by every octahedron in the layer as a cheap linked duplicate.
        octa_mesh = make_octahedron_mesh(f"{name}_octa", CFG.OCTA_RADIUS)
        octa_mesh.materials.append(materials[name])

        for ix in range(CFG.GRID_X):
            for iy in range(CFG.GRID_Y):
                # Ring index = distance to the nearest grid edge. Border cells
                # are ring 0 (erode first); each step inward is the next shell.
                ring = min(ix, CFG.GRID_X - 1 - ix,
                           iy, CFG.GRID_Y - 1 - iy)
                for kz in range(height):
                    loc = (x0 + ix * step_xy,
                           y0 + iy * step_xy,
                           z_cursor + CFG.OCTA_RADIUS + kz * step_z)
                    obj = bpy.data.objects.new(
                        f"{name}_{ix}_{iy}_{kz}", octa_mesh)
                    obj.location = loc
                    obj["erosion_ring"] = ring
                    coll.objects.link(obj)
                    objects[name].append(obj)

        layer_top = z_cursor + (height - 1) * step_z + 2 * CFG.OCTA_RADIUS
        bounds[name] = (z_cursor, layer_top)
        z_cursor = layer_top + CFG.LAYER_SEPARATION_Z

    return {
        "objects": objects,
        "bounds": bounds,
        "lateral_extent": max(CFG.GRID_X, CFG.GRID_Y) * step_xy,
    }


# =============================================================================
# Animation helpers
# =============================================================================

def _iter_fcurves(obj):
    """Yield an object's F-Curves across Blender versions. Pre-4.4 uses
    action.fcurves; 4.4+/5.x 'slotted actions' keep them in
    layers -> strips -> channelbags."""
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    action = ad.action
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        for fc in legacy:
            yield fc
        return
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for cbag in getattr(strip, "channelbags", []):
                for fc in cbag.fcurves:
                    yield fc


def _set_interpolation(obj, interp="SINE", easing="EASE_OUT"):
    for fc in _iter_fcurves(obj):
        for kp in fc.keyframe_points:
            kp.interpolation = interp
            kp.easing = easing


def _rand_unit(rng):
    """A uniform-ish random unit vector (rejection sampled in the unit ball)."""
    while True:
        v = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        if 0.05 < v.length <= 1.0:
            return v.normalized()


# =============================================================================
# Animation: sacrificial erosion, octahedron by octahedron (edge -> center)
# =============================================================================

def animate_erosion(sacrificial_objs, rng):
    """Erode the sacrificial layer strictly ring-by-ring, outside -> in, and
    return the frame at which the last octahedron has fully dissipated."""
    if not sacrificial_objs:
        return CFG.EROSION_START

    last_end = CFG.EROSION_START
    for obj in sacrificial_objs:
        base_loc = obj.location.copy()
        base_scale = obj.scale.copy()
        ring = int(obj.get("erosion_ring", 0))

        # Outward (radial) component in XY, plus a big random shove.
        radial = Vector((base_loc.x, base_loc.y, 0.0))
        if radial.length < 1e-4:
            radial = _rand_unit(rng)
        radial.normalize()
        direction = (radial * CFG.EROSION_RADIAL_STRENGTH
                     + _rand_unit(rng) * CFG.EROSION_AGITATION)
        direction.z += CFG.EROSION_LIFT_BIAS
        if direction.length < 1e-4:
            direction = radial
        direction.normalize()

        # Each ring gets its own window; the within-ring stagger is bounded so
        # a piece can never erode before its outer ring is completely gone.
        ring_start = CFG.EROSION_START + ring * CFG.EROSION_RING_INTERVAL
        start = int(ring_start + rng.uniform(0.0, CFG.EROSION_RING_STAGGER))
        end = start + CFG.EROSION_PIECE_DURATION
        last_end = max(last_end, end)

        target = base_loc + direction * CFG.EROSION_STREAM_DISTANCE

        obj.location = base_loc
        obj.keyframe_insert("location", frame=start)
        obj.location = target
        obj.keyframe_insert("location", frame=end)

        obj.scale = base_scale
        obj.keyframe_insert("scale", frame=start)
        obj.scale = base_scale * 0.01
        obj.keyframe_insert("scale", frame=end)

        spin = CFG.EROSION_SPIN
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.keyframe_insert("rotation_euler", frame=start)
        obj.rotation_euler = (spin * rng.uniform(-1, 1),
                              spin * rng.uniform(-1, 1),
                              spin * rng.uniform(-1, 1))
        obj.keyframe_insert("rotation_euler", frame=end)

        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert("hide_viewport", frame=end - 1)
        obj.keyframe_insert("hide_render", frame=end - 1)
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=end)
        obj.keyframe_insert("hide_render", frame=end)

        _set_interpolation(obj, "SINE", "EASE_OUT")

    return last_end


# =============================================================================
# Animation: water molecules
# =============================================================================

def build_and_animate_water(water_mesh, water_mat, geom, rng, erosion_end):
    coll = ensure_collection("Water")
    water_mesh.materials.append(water_mat)

    bounds = geom["bounds"]
    sac_z_min, sac_z_max = bounds["sacrificial"]
    sac_center_z = 0.5 * (sac_z_min + sac_z_max)
    ring_radius = 0.5 * geom["lateral_extent"] + CFG.WATER_SPAWN_RING_MARGIN

    for i in range(CFG.WATER_COUNT):
        ang = rng.uniform(0, 2 * math.pi)
        r = ring_radius * rng.uniform(0.9, 1.15)
        spawn = Vector((r * math.cos(ang), r * math.sin(ang),
                        sac_center_z + rng.uniform(-CFG.WATER_SPAWN_Z_SPREAD,
                                                   CFG.WATER_SPAWN_Z_SPREAD)))

        tang = rng.uniform(0, 2 * math.pi)
        tr = 0.5 * geom["lateral_extent"] * rng.uniform(0.2, 1.0)
        target = Vector((tr * math.cos(tang), tr * math.sin(tang),
                         rng.uniform(sac_z_min, sac_z_max)))

        obj = bpy.data.objects.new(f"water_{i}", water_mesh)
        obj.location = spawn
        coll.objects.link(obj)

        appear = int(rng.uniform(CFG.WATER_FADE_IN_START, CFG.WATER_FADE_IN_END))
        arrive = int(rng.uniform(CFG.WATER_APPROACH_END - 12,
                                 CFG.WATER_APPROACH_END))
        # Water lingers until erosion is finished, then washes away.
        gone_lo = min(erosion_end, CFG.WATER_DISSIPATE_END - 5)
        gone = int(rng.uniform(gone_lo, CFG.WATER_DISSIPATE_END))

        full = Vector((1.0, 1.0, 1.0))
        tiny = Vector((0.001, 0.001, 0.001))

        # Scale: fade in, hold, dissipate
        obj.scale = tiny
        obj.keyframe_insert("scale", frame=appear)
        obj.scale = full
        obj.keyframe_insert("scale", frame=min(appear + 8, arrive))
        obj.scale = full
        obj.keyframe_insert("scale", frame=arrive)
        obj.scale = tiny
        obj.keyframe_insert("scale", frame=gone)

        # Location: approach, then wash outward while dissipating
        obj.location = spawn
        obj.keyframe_insert("location", frame=appear)
        obj.location = target
        obj.keyframe_insert("location", frame=arrive)
        drift = target + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1),
                                 0.5)) * 1.5
        obj.location = drift
        obj.keyframe_insert("location", frame=gone)

        # Hide after dissipation
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert("hide_viewport", frame=gone - 1)
        obj.keyframe_insert("hide_render", frame=gone - 1)
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=gone)
        obj.keyframe_insert("hide_render", frame=gone)

        _set_interpolation(obj, "SINE", "EASE_IN_OUT")


# =============================================================================
# Camera, lights, world, render settings
# =============================================================================

def _point_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_camera_and_lights(geom):
    extent = geom["lateral_extent"]
    top_z = geom["bounds"]["film"][1]
    sac = geom["bounds"]["sacrificial"]
    look_at = Vector((0.0, 0.0, 0.5 * (sac[0] + sac[1])))
    dist = extent * 2.2 + 6.0

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (dist, -dist, top_z * 0.5 + extent * 0.9)
    bpy.context.scene.collection.objects.link(cam)
    _point_at(cam, look_at)
    bpy.context.scene.camera = cam

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.location = (dist, -dist, dist)
    _point_at(sun, look_at)
    bpy.context.scene.collection.objects.link(sun)

    area_data = bpy.data.lights.new("Fill", type="AREA")
    area_data.energy = 400.0
    area_data.size = extent * 2.0
    area = bpy.data.objects.new("Fill", area_data)
    area.location = (-dist, dist * 0.5, top_z + extent)
    _point_at(area, look_at)
    bpy.context.scene.collection.objects.link(area)

    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.02, 0.02, 0.03, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def _available_engines():
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        return {item.identifier for item in prop.enum_items}
    except Exception:
        return set()


def _set_viewport_shading():
    if not CFG.SET_MATERIAL_SHADING:
        return
    try:
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.type = "MATERIAL"
    except Exception:
        pass


def setup_render_and_timeline():
    scene = bpy.context.scene
    scene.frame_start = CFG.FRAME_START
    scene.frame_end = CFG.FRAME_END
    scene.render.fps = CFG.FPS
    scene.render.resolution_x = CFG.RESOLUTION_X
    scene.render.resolution_y = CFG.RESOLUTION_Y
    scene.render.resolution_percentage = CFG.RESOLUTION_PERCENT

    # Transparent background (alpha) for the final render.
    scene.render.film_transparent = bool(CFG.TRANSPARENT_BG)

    if CFG.USE_EEVEE:
        engines = _available_engines()
        for candidate in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            if candidate in engines:
                scene.render.engine = candidate
                break
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            for attr in ("taa_render_samples", "taa_samples"):
                if hasattr(eevee, attr):
                    setattr(eevee, attr, CFG.EEVEE_SAMPLES)
            # Bloom makes the emissive colors glow (available on EEVEE Legacy).
            if hasattr(eevee, "use_bloom"):
                eevee.use_bloom = True

    if CFG.USE_SIMPLIFY:
        scene.render.use_simplify = True
        scene.render.simplify_subdivision = CFG.SIMPLIFY_VIEWPORT_SUBDIV

    _set_viewport_shading()


# =============================================================================
# Main
# =============================================================================

def main():
    rng = random.Random(CFG.RANDOM_SEED)

    if CFG.CLEAR_SCENE:
        clear_scene()

    materials = {
        "substrate":   make_material("Substrate", CFG.COLOR_SUBSTRATE,
                                     CFG.EMISSION_STRENGTH),
        "sacrificial": make_material("Sacrificial", CFG.COLOR_SACRIFICIAL,
                                     CFG.EMISSION_STRENGTH),
        "film":        make_material("Film", CFG.COLOR_FILM,
                                     CFG.EMISSION_STRENGTH),
    }
    water_mat = make_material("Water", CFG.COLOR_WATER,
                              emission_strength=0.3, alpha=CFG.WATER_ALPHA)
    water_mesh = make_uv_sphere_mesh("WaterMolecule", CFG.WATER_RADIUS,
                                     CFG.WATER_UV_SEGMENTS)

    geom = build_layers(materials)
    erosion_end = animate_erosion(geom["objects"]["sacrificial"], rng)
    build_and_animate_water(water_mesh, water_mat, geom, rng, erosion_end)

    setup_camera_and_lights(geom)
    setup_render_and_timeline()

    # Make sure the timeline covers erosion + water dissipation.
    needed_end = max(CFG.FRAME_END, erosion_end + 5, CFG.WATER_DISSIPATE_END + 5)
    if bpy.context.scene.frame_end < needed_end:
        bpy.context.scene.frame_end = needed_end

    bpy.context.scene.frame_set(CFG.FRAME_START)

    total = sum(len(v) for v in geom["objects"].values())
    print("[membrane_liftoff] Built {} octahedra (one object each) + {} water "
          "molecules. Frames {}-{} @ {}fps. Transparent BG: {}."
          .format(total, CFG.WATER_COUNT, CFG.FRAME_START, CFG.FRAME_END,
                  CFG.FPS, CFG.TRANSPARENT_BG))


if __name__ == "__main__":
    main()


# =============================================================================
# HOW TO SAVE THE VIDEO
# =============================================================================
#
# After running this script, pick ONE of the workflows below.
#
# ---------------------------------------------------------------------------
# A) Quick MP4 (no transparency -- background comes out solid/black)
# ---------------------------------------------------------------------------
#   1. Properties editor -> Output tab (the printer icon).
#   2. Set "Frame Range" (already set by the script: 1..165).
#   3. Under "Output", choose a folder and set File Format = "FFmpeg Video".
#   4. Expand "Encoding": Container = "MPEG-4", Video Codec = "H.264".
#      (Output Quality "High" is a good default.)
#   5. Top menu -> Render -> Render Animation  (or press Ctrl+F12).
#   6. Blender writes a single .mp4 into the folder you chose.
#
#   NOTE: MP4/H.264 has no alpha channel, so TRANSPARENT_BG has no visible
#   effect here -- you'll get a solid background. Use B) or C) to keep it.
#
# ---------------------------------------------------------------------------
# B) Transparent video (WebM / VP9 -- keeps the alpha background)
# ---------------------------------------------------------------------------
#   1. Keep TRANSPARENT_BG = True (default).
#   2. Output tab -> File Format = "FFmpeg Video".
#   3. Encoding: Container = "WebM", Video Codec = "WEBM / VP9".
#   4. Set "Color" to "RGBA" (this is what carries the alpha).
#   5. Render -> Render Animation.
#   -> The .webm plays with a transparent background in apps that support it.
#
# ---------------------------------------------------------------------------
# C) Best quality / most compatible: PNG sequence (RGBA) -> video
# ---------------------------------------------------------------------------
#   1. Keep TRANSPARENT_BG = True.
#   2. Output tab -> File Format = "PNG", Color = "RGBA".
#   3. Set the output path to a folder ending in a name prefix, e.g.
#         //frames/liftoff_
#      (the // means "relative to this .blend file").
#   4. Render -> Render Animation. You get liftoff_0001.png, 0002.png, ...
#      each with a transparent background.
#   5. Combine them into a video with FFmpeg (transparency preserved as VP9):
#         ffmpeg -framerate 24 -i frames/liftoff_%04d.png \
#                -c:v libvpx-vp9 -pix_fmt yuva420p liftoff.webm
#      ...or make a normal (opaque) MP4 over a chosen background color:
#         ffmpeg -framerate 24 -i frames/liftoff_%04d.png \
#                -c:v libx264 -pix_fmt yuv420p liftoff.mp4
#
# TIP: To preview timing without a full render, use
#      View menu -> Viewport Render Animation in the 3D viewport.
# =============================================================================
