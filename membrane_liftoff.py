"""
membrane_liftoff.py
===================

Blender Python (bpy) script that animates the dissolution of a sacrificial
layer during membrane liftoff.

Scene layout (bottom -> top along +Z):
    * Substrate      (thickest stack of octahedra)
    * Sacrificial    (middle stack -- this is the layer that erodes away)
    * Top film       (the "membrane" -- stays in place, does not fall)

Water molecules (small spheres, ~2 octahedra across) fade in around the
edges, stream toward the sacrificial layer, and drive the erosion: the
sacrificial octahedra loosen, then stream out sideways from the layer and
dissipate (they do NOT drop). The water dissipates afterward.

HOW TO RUN
----------
1. Open Blender (tested against 3.x / 4.x).
2. Open the Scripting workspace, load this file, press "Run Script".
   (Or from a terminal:  blender --python membrane_liftoff.py )
3. Press the spacebar / play button to watch the animation in the viewport.

Everything you might want to tweak lives in the CONFIG block below. Nothing
about the geometry, sizing, counts, or timing is hard-coded elsewhere -- if a
value matters, it is a named parameter here so you can edit it in one place.

NOTE ON "STACKS": each layer shares the same lateral grid of octahedra
(GRID_X * GRID_Y columns). The 3 / 4 / 5 "stacks" numbers are interpreted as
how many octahedra TALL each layer is, so the substrate is the thickest and
the film is the thinnest. If you meant something else by "stacks", change
LAYER_HEIGHTS below.
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
    # Spacing between octahedron centers. With gap = 0 the tips just touch.
    OCTA_GAP_XY = 0.05              # extra horizontal gap between columns
    OCTA_GAP_Z = 0.05              # extra vertical gap within a stack

    # ---- Lateral extent of every layer (shared grid of columns) --------------
    GRID_X = 6                      # octahedra columns along X
    GRID_Y = 6                      # octahedra columns along Y

    # ---- Layer thicknesses, in octahedra tall (the "stacks" numbers) ---------
    # Order is bottom -> top.
    LAYER_HEIGHTS = {
        "substrate":   5,          # thickest
        "sacrificial": 4,          # the layer that erodes
        "film":        3,          # the membrane on top
    }
    # Small vertical gap between adjacent layers so they read as distinct.
    LAYER_SEPARATION_Z = 0.15

    # ---- Water molecules -----------------------------------------------------
    WATER_COUNT = 40                # how many water "molecules" to spawn
    WATER_RADIUS = 2.0 * OCTA_RADIUS  # ~ two octahedra across (a molecule "blob")
    WATER_UV_SUBDIV = 12            # sphere resolution (keep low for speed)
    # Where water spawns: a ring/shell around the sacrificial layer.
    WATER_SPAWN_RING_MARGIN = 3.0   # how far out from the stack water starts
    WATER_SPAWN_Z_SPREAD = 2.0      # vertical spread of the spawn shell

    # ---- Colors (RGBA, linear-ish) -------------------------------------------
    COLOR_SUBSTRATE   = (0.35, 0.35, 0.40, 1.0)   # grey base
    COLOR_SACRIFICIAL = (0.85, 0.55, 0.20, 1.0)   # warm orange, stands out
    COLOR_FILM        = (0.30, 0.55, 0.85, 1.0)   # blue membrane
    COLOR_WATER       = (0.20, 0.70, 0.95, 1.0)   # cyan, semi-transparent
    WATER_ALPHA       = 0.45

    # ---- Timeline (frames) ---------------------------------------------------
    FPS = 24
    FRAME_START = 1
    # Phase boundaries. Water fades in and approaches, then erosion overlaps
    # and finishes, then the water dissipates.
    WATER_FADE_IN_START   = 1
    WATER_FADE_IN_END     = 20
    WATER_APPROACH_END    = 60      # water has reached the sacrificial layer
    EROSION_START         = 45      # erosion begins while water is still arriving
    EROSION_END           = 150     # all sacrificial octahedra gone by here
    WATER_DISSIPATE_END   = 175     # water fully faded out
    FRAME_END             = 185     # a little tail for settling

    # ---- Erosion motion ------------------------------------------------------
    EROSION_STREAM_DISTANCE = 6.0   # how far octahedra drift outward before vanish
    EROSION_STREAM_LIFT = 0.4       # slight upward drift so it "streams", not sinks
    EROSION_SPIN = 4.0              # radians of tumble as a piece streams out
    EROSION_PIECE_DURATION = 30     # frames from loosen -> fully dissipated

    # ---- Rendering / viewport smoothness ------------------------------------
    USE_EEVEE = True                # EEVEE is far faster than Cycles for playback
    EEVEE_SAMPLES = 16              # low samples = smooth realtime playback
    RESOLUTION_X = 1280
    RESOLUTION_Y = 720
    RESOLUTION_PERCENT = 100
    USE_SIMPLIFY = True             # cap subdivisions during playback
    SIMPLIFY_VIEWPORT_SUBDIV = 1


# =============================================================================
# Utility helpers
# =============================================================================

def clear_scene():
    """Remove all objects / orphan meshes / materials for a clean start."""
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves,
                 bpy.data.lights, bpy.data.cameras):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def make_octahedron_mesh(radius):
    """Create and return a single octahedron mesh datablock (6 verts, 8 faces).

    The mesh is created once and shared by every octahedron object (linked
    duplicates) so thousands of pieces stay light on memory and fast to draw.
    """
    r = radius
    verts = [
        (0, 0,  r), (0, 0, -r),   # top, bottom
        ( r, 0, 0), (-r, 0, 0),   # +x, -x
        (0,  r, 0), (0, -r, 0),   # +y, -y
    ]
    faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),   # upper 4 faces
        (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),   # lower 4 faces
    ]
    mesh = bpy.data.meshes.new("Octahedron")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    # Smooth-ish look is undesirable for faceted octahedra; keep flat shading.
    return mesh


def make_material(name, rgba, alpha=1.0, emission_boost=0.0):
    """Simple principled material. If alpha < 1, set up blended transparency."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        # 'Roughness' exists across versions.
        bsdf.inputs["Roughness"].default_value = 0.5
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if emission_boost > 0.0:
            # Emission input names differ across Blender versions.
            for key in ("Emission Color", "Emission"):
                if key in bsdf.inputs:
                    bsdf.inputs[key].default_value = rgba
                    break
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_boost
    if alpha < 1.0:
        mat.blend_method = "BLEND"
        # 'show_transparent_back' off looks cleaner for blobs.
        if hasattr(mat, "show_transparent_back"):
            mat.show_transparent_back = False
    return mat


def new_linked_object(name, mesh, material, collection, location):
    """Create an object that shares `mesh` (linked duplicate) at `location`."""
    obj = bpy.data.objects.new(name, mesh)
    if material is not None and len(obj.data.materials) == 0:
        # Assign material to the shared mesh only once; linked dups inherit it.
        if material.name not in [m.name for m in mesh.materials]:
            mesh.materials.append(material)
    obj.location = location
    collection.objects.link(obj)
    return obj


def ensure_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


# =============================================================================
# Build the stacked layers
# =============================================================================

def build_layers(octa_mesh, materials):
    """Create the three layers of octahedra.

    Returns a dict with lists of objects per layer and geometry metadata used
    later for the water + erosion animation.
    """
    step_xy = 2.0 * CFG.OCTA_RADIUS + CFG.OCTA_GAP_XY
    step_z = 2.0 * CFG.OCTA_RADIUS + CFG.OCTA_GAP_Z

    # Center the lateral grid on the origin.
    x0 = -0.5 * (CFG.GRID_X - 1) * step_xy
    y0 = -0.5 * (CFG.GRID_Y - 1) * step_xy

    layer_order = ["substrate", "sacrificial", "film"]  # bottom -> top

    collections = {name: ensure_collection(f"Layer_{name}") for name in layer_order}
    objects = {name: [] for name in layer_order}

    z_cursor = 0.0
    layer_bounds = {}  # name -> (z_min, z_max)

    for name in layer_order:
        height = CFG.LAYER_HEIGHTS[name]
        mat = materials[name]
        z_min = z_cursor
        for kz in range(height):
            z = z_cursor + kz * step_z + CFG.OCTA_RADIUS
            for ix in range(CFG.GRID_X):
                for iy in range(CFG.GRID_Y):
                    loc = (x0 + ix * step_xy, y0 + iy * step_xy, z)
                    obj = new_linked_object(
                        f"{name}_{ix}_{iy}_{kz}", octa_mesh, mat,
                        collections[name], loc)
                    objects[name].append(obj)
        z_max = z_cursor + (height - 1) * step_z + 2 * CFG.OCTA_RADIUS
        layer_bounds[name] = (z_min, z_max)
        z_cursor = z_max + CFG.LAYER_SEPARATION_Z

    lateral_extent = max(CFG.GRID_X, CFG.GRID_Y) * step_xy
    return {
        "objects": objects,
        "collections": collections,
        "bounds": layer_bounds,
        "step_xy": step_xy,
        "step_z": step_z,
        "lateral_extent": lateral_extent,
    }


# =============================================================================
# Animation: sacrificial layer erosion
# =============================================================================

def animate_erosion(sacrificial_objs, rng):
    """Each sacrificial octahedron loosens, streams outward from the nearest
    side, tumbles, shrinks to nothing, then is hidden. Nothing falls."""
    n = len(sacrificial_objs)
    if n == 0:
        return

    for idx, obj in enumerate(sacrificial_objs):
        base_loc = obj.location.copy()
        base_scale = obj.scale.copy()

        # Outward direction in XY only (stream sideways, not down).
        radial = Vector((base_loc.x, base_loc.y, 0.0))
        if radial.length < 1e-4:
            angle = rng.uniform(0, 2 * math.pi)
            radial = Vector((math.cos(angle), math.sin(angle), 0.0))
        radial.normalize()

        # Stagger start times so the layer erodes progressively.
        jitter = rng.uniform(0.0, 1.0)
        span = max(1, CFG.EROSION_END - CFG.EROSION_START - CFG.EROSION_PIECE_DURATION)
        start = int(CFG.EROSION_START + jitter * span)
        end = start + CFG.EROSION_PIECE_DURATION

        target = base_loc + radial * CFG.EROSION_STREAM_DISTANCE
        target.z += CFG.EROSION_STREAM_LIFT * rng.uniform(0.5, 1.5)

        # --- Location keyframes ---
        obj.location = base_loc
        obj.keyframe_insert("location", frame=start)
        obj.location = target
        obj.keyframe_insert("location", frame=end)

        # --- Scale: full -> ~0 (dissipate) ---
        obj.scale = base_scale
        obj.keyframe_insert("scale", frame=start)
        obj.scale = base_scale * 0.01
        obj.keyframe_insert("scale", frame=end)

        # --- Tumble as it streams out ---
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.keyframe_insert("rotation_euler", frame=start)
        spin = CFG.EROSION_SPIN * rng.uniform(0.5, 1.5)
        ax = rng.uniform(-1, 1)
        ay = rng.uniform(-1, 1)
        obj.rotation_euler = (spin * ax, spin * ay, spin * rng.uniform(-1, 1))
        obj.keyframe_insert("rotation_euler", frame=end)

        # --- Hide once dissipated (viewport + render) ---
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert("hide_viewport", frame=end - 1)
        obj.keyframe_insert("hide_render", frame=end - 1)
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=end)
        obj.keyframe_insert("hide_render", frame=end)

        # Ease the motion so streaming looks fluid.
        _set_interpolation(obj, "SINE", "EASE_OUT")


def _set_interpolation(obj, interp="SINE", easing="EASE_IN_OUT"):
    if not obj.animation_data or not obj.animation_data.action:
        return
    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = interp
            kp.easing = easing


# =============================================================================
# Animation: water molecules
# =============================================================================

def build_and_animate_water(water_mesh, water_mat, geom, rng):
    coll = ensure_collection("Water")
    objects = geom["objects"]
    bounds = geom["bounds"]

    sac_z_min, sac_z_max = bounds["sacrificial"]
    sac_center_z = 0.5 * (sac_z_min + sac_z_max)
    ring_radius = 0.5 * geom["lateral_extent"] + CFG.WATER_SPAWN_RING_MARGIN

    for i in range(CFG.WATER_COUNT):
        # Spawn on a shell around the sacrificial layer.
        ang = rng.uniform(0, 2 * math.pi)
        r = ring_radius * rng.uniform(0.9, 1.15)
        sx = r * math.cos(ang)
        sy = r * math.sin(ang)
        sz = sac_center_z + rng.uniform(-CFG.WATER_SPAWN_Z_SPREAD,
                                        CFG.WATER_SPAWN_Z_SPREAD)
        spawn = Vector((sx, sy, sz))

        # Target a point inside/at the edge of the sacrificial layer.
        tang = rng.uniform(0, 2 * math.pi)
        tr = 0.5 * geom["lateral_extent"] * rng.uniform(0.2, 1.0)
        tx = tr * math.cos(tang)
        ty = tr * math.sin(tang)
        tz = rng.uniform(sac_z_min, sac_z_max)
        target = Vector((tx, ty, tz))

        obj = bpy.data.objects.new(f"water_{i}", water_mesh)
        if water_mat.name not in [m.name for m in water_mesh.materials]:
            water_mesh.materials.append(water_mat)
        obj.location = spawn
        coll.objects.link(obj)

        # Timing: fade in, then approach, then dissipate.
        appear = int(rng.uniform(CFG.WATER_FADE_IN_START, CFG.WATER_FADE_IN_END))
        arrive = int(rng.uniform(CFG.WATER_APPROACH_END - 12, CFG.WATER_APPROACH_END))
        gone = int(rng.uniform(CFG.EROSION_END, CFG.WATER_DISSIPATE_END))

        full_scale = Vector((1.0, 1.0, 1.0))
        tiny = Vector((0.001, 0.001, 0.001))

        # Scale: 0 -> full (fade in) at spawn, hold, then -> 0 (dissipate).
        obj.scale = tiny
        obj.keyframe_insert("scale", frame=appear)
        obj.scale = full_scale
        obj.keyframe_insert("scale", frame=min(appear + 8, arrive))
        obj.scale = full_scale
        obj.keyframe_insert("scale", frame=arrive)
        obj.scale = tiny
        obj.keyframe_insert("scale", frame=gone)

        # Location: spawn -> target (approach), then gentle outward drift while
        # dissipating so it looks like it's washing away.
        obj.location = spawn
        obj.keyframe_insert("location", frame=appear)
        obj.location = target
        obj.keyframe_insert("location", frame=arrive)
        drift = target + Vector(
            (rng.uniform(-1, 1), rng.uniform(-1, 1), 0.5)) * 1.5
        obj.location = drift
        obj.keyframe_insert("location", frame=gone)

        # Hide after dissipation.
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

def setup_camera_and_lights(geom):
    extent = geom["lateral_extent"]
    top_z = geom["bounds"]["film"][1]
    center_z = 0.5 * top_z

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    dist = extent * 2.2 + 6.0
    cam.location = (dist, -dist, center_z + extent * 0.9)
    bpy.context.scene.collection.objects.link(cam)
    # Aim camera at the sacrificial layer center.
    target = Vector((0.0, 0.0, 0.5 * (geom["bounds"]["sacrificial"][0]
                                      + geom["bounds"]["sacrificial"][1])))
    _point_at(cam, target)
    bpy.context.scene.camera = cam

    # Key light (sun)
    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.location = (dist, -dist, dist)
    _point_at(sun, Vector((0, 0, center_z)))
    bpy.context.scene.collection.objects.link(sun)

    # Fill area light
    area_data = bpy.data.lights.new("Fill", type="AREA")
    area_data.energy = 400.0
    area_data.size = extent * 2.0
    area = bpy.data.objects.new("Fill", area_data)
    area.location = (-dist, dist * 0.5, center_z + extent)
    _point_at(area, Vector((0, 0, center_z)))
    bpy.context.scene.collection.objects.link(area)

    # World background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.02, 0.02, 0.03, 1.0)
        bg.inputs["Strength"].default_value = 1.0


def _point_at(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_render_and_timeline():
    scene = bpy.context.scene
    scene.frame_start = CFG.FRAME_START
    scene.frame_end = CFG.FRAME_END
    scene.render.fps = CFG.FPS
    scene.render.resolution_x = CFG.RESOLUTION_X
    scene.render.resolution_y = CFG.RESOLUTION_Y
    scene.render.resolution_percentage = CFG.RESOLUTION_PERCENT

    # Pick the EEVEE engine name available in this Blender version.
    if CFG.USE_EEVEE:
        engines = _available_engines()
        if "BLENDER_EEVEE_NEXT" in engines:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        elif "BLENDER_EEVEE" in engines:
            scene.render.engine = "BLENDER_EEVEE"
        # Configure samples if the eevee settings block exists.
        eevee = getattr(scene, "eevee", None)
        if eevee is not None and hasattr(eevee, "taa_render_samples"):
            eevee.taa_render_samples = CFG.EEVEE_SAMPLES
            eevee.taa_samples = CFG.EEVEE_SAMPLES

    # Simplify keeps playback smooth by capping subdivision work.
    if CFG.USE_SIMPLIFY:
        scene.render.use_simplify = True
        scene.render.simplify_subdivision = CFG.SIMPLIFY_VIEWPORT_SUBDIV

    # Try to set solid/material viewport shading for smoother scrubbing.
    _set_viewport_shading()


def _available_engines():
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        return {item.identifier for item in prop.enum_items}
    except Exception:
        return {"BLENDER_EEVEE"}


def _set_viewport_shading():
    try:
        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
    except Exception:
        pass  # No UI context (e.g. background mode) -- ignore.


# =============================================================================
# Main
# =============================================================================

def main():
    rng = random.Random(CFG.RANDOM_SEED)

    if CFG.CLEAR_SCENE:
        clear_scene()

    # Shared geometry + materials
    octa_mesh = make_octahedron_mesh(CFG.OCTA_RADIUS)
    water_mesh = _make_uv_sphere_mesh(CFG.WATER_RADIUS, CFG.WATER_UV_SUBDIV)

    materials = {
        "substrate":   make_material("Substrate", CFG.COLOR_SUBSTRATE),
        "sacrificial": make_material("Sacrificial", CFG.COLOR_SACRIFICIAL),
        "film":        make_material("Film", CFG.COLOR_FILM),
    }
    water_mat = make_material("Water", CFG.COLOR_WATER,
                              alpha=CFG.WATER_ALPHA, emission_boost=0.3)

    geom = build_layers(octa_mesh, materials)

    animate_erosion(geom["objects"]["sacrificial"], rng)
    build_and_animate_water(water_mesh, water_mat, geom, rng)

    setup_camera_and_lights(geom)
    setup_render_and_timeline()

    bpy.context.scene.frame_set(CFG.FRAME_START)
    print("[membrane_liftoff] Scene built. "
          f"{sum(len(v) for v in geom['objects'].values())} octahedra, "
          f"{CFG.WATER_COUNT} water molecules. "
          f"Frames {CFG.FRAME_START}-{CFG.FRAME_END} @ {CFG.FPS}fps.")


def _make_uv_sphere_mesh(radius, subdiv):
    """Build a low-poly UV sphere mesh via bmesh (no operator/context needed)."""
    mesh = bpy.data.meshes.new("WaterMolecule")
    bm = bmesh.new()
    seg = max(6, subdiv)
    ring = max(4, subdiv // 2)
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=ring, radius=radius)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


if __name__ == "__main__":
    main()
