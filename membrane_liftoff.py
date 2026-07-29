"""
membrane_liftoff.py
===================

Blender Python (bpy) script that animates the dissolution of a sacrificial
layer during membrane liftoff.  Written for / tested against Blender 4.x-5.x.

Scene layout (bottom -> top along +Z):
    * Substrate      (thickest stack of octahedra)
    * Sacrificial    (middle stack -- this is the layer that erodes away)
    * Top film       (the "membrane" -- stays in place, does not fall)

Each layer is a lateral grid (GRID_X x GRID_Y) of *stacks*.  A "stack" is a
vertical column of octahedra, and -- per request -- each stack is ONE Blender
object (all its octahedra joined into a single mesh).  That keeps the scene
light and lets whole stacks erode as units.

Water molecules (small spheres, ~2 octahedra across) fade in around the edges,
stream toward the sacrificial layer, and drive the erosion.  Erosion begins at
the OUTER EDGES of the sacrificial layer and works inward to the center: each
stack loosens, streams out sideways (it does NOT drop), tumbles, shrinks, and
disappears.  The water then dissipates.

HOW TO RUN
----------
1. Open Blender, go to the Scripting workspace, load this file, "Run Script".
   (Or:  blender --python membrane_liftoff.py )
2. Press play to watch it in the viewport.  The script switches 3D viewports to
   Material shading so the colors show up immediately.

Everything tweakable lives in the CONFIG block below -- geometry, sizes, counts,
colors, and the full frame timeline are all named parameters in one place.

NOTE ON "STACKS": the 3 / 4 / 5 numbers are how many octahedra TALL each layer
is (substrate thickest, film thinnest), over a shared GRID_X x GRID_Y footprint.
If you meant something else, edit LAYER_HEIGHTS / GRID_X / GRID_Y below.
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

    # ---- Colors (RGBA) -------------------------------------------------------
    COLOR_SUBSTRATE   = (0.35, 0.35, 0.40, 1.0)   # grey base
    COLOR_SACRIFICIAL = (0.90, 0.45, 0.12, 1.0)   # warm orange, stands out
    COLOR_FILM        = (0.20, 0.55, 0.90, 1.0)   # blue membrane
    COLOR_WATER       = (0.20, 0.70, 0.95, 1.0)   # cyan, semi-transparent
    WATER_ALPHA       = 0.45

    # ---- Timeline (frames) ---------------------------------------------------
    FPS = 24
    FRAME_START = 1
    WATER_FADE_IN_START   = 1
    WATER_FADE_IN_END     = 20
    WATER_APPROACH_END    = 60      # water has reached the sacrificial layer
    EROSION_START         = 45      # erosion begins (edges first)
    EROSION_END           = 150     # all sacrificial stacks gone by here
    WATER_DISSIPATE_END   = 175     # water fully faded out
    FRAME_END             = 185

    # ---- Erosion motion ------------------------------------------------------
    EROSION_STREAM_DISTANCE = 6.0   # how far a stack drifts outward before vanish
    EROSION_STREAM_LIFT = 0.4       # slight upward drift so it streams, not sinks
    EROSION_SPIN = 3.0              # radians of tumble as a stack streams out
    EROSION_PIECE_DURATION = 28     # frames from loosen -> fully dissipated
    EROSION_JITTER = 6              # +/- frames of randomness on top of edge order

    # ---- Rendering / viewport smoothness ------------------------------------
    USE_EEVEE = True                # EEVEE is far faster than Cycles for playback
    EEVEE_SAMPLES = 16              # low samples = smooth realtime playback
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
    """Remove all objects and datablocks WITHOUT relying on operator context.

    Operator-based clearing (bpy.ops) is fragile when run from the Scripting
    workspace and can leave stragglers behind (producing '.001' name clashes on
    the next run).  Direct datablock removal is context-independent.
    """
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


def _add_octahedron(bm, cx, cy, cz, r):
    """Append one octahedron (6 verts, 8 faces) centered at (cx,cy,cz)."""
    v = [
        bm.verts.new((cx,     cy,     cz + r)),
        bm.verts.new((cx,     cy,     cz - r)),
        bm.verts.new((cx + r, cy,     cz)),
        bm.verts.new((cx - r, cy,     cz)),
        bm.verts.new((cx,     cy + r, cz)),
        bm.verts.new((cx,     cy - r, cz)),
    ]
    for a, b, c in _OCTA_FACES:
        bm.faces.new((v[a], v[b], v[c]))


def make_column_mesh(name, height, radius, step_z):
    """One joined mesh = a vertical stack of `height` octahedra.

    Local origin sits at the base of the column (z=0 is the bottom tip plane),
    so an object placed at world z = layer_base sits correctly on the layer.
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    for k in range(height):
        cz = radius + k * step_z
        _add_octahedron(bm, 0.0, 0.0, cz, radius)
    bm.to_mesh(mesh)
    bm.free()
    # Faceted (flat) look for the octahedra.
    for poly in mesh.polygons:
        poly.use_smooth = False
    mesh.update()
    return mesh


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


def make_material(name, rgba, alpha=1.0, emission_boost=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.5
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if emission_boost > 0.0:
            for key in ("Emission Color", "Emission"):   # renamed in 4.0
                if key in bsdf.inputs:
                    bsdf.inputs[key].default_value = rgba
                    break
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_boost
    # Viewport display color (so it reads even in Solid shading).
    mat.diffuse_color = rgba
    if alpha < 1.0:
        for attr, val in (("blend_method", "BLEND"),
                          ("show_transparent_back", False)):
            if hasattr(mat, attr):
                setattr(mat, attr, val)
    return mat


def new_stack_object(name, mesh, collection, location):
    """Create an object sharing `mesh` (linked duplicate) at `location`.
    Material lives on the shared mesh, which is safe here because every stack
    sharing a mesh belongs to the SAME layer (one color)."""
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    collection.objects.link(obj)
    return obj


# =============================================================================
# Build the stacked layers
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

        # One mesh per layer -> one material -> correct per-layer color, and
        # every stack in the layer shares it as a cheap linked duplicate.
        col_mesh = make_column_mesh(f"{name}_stack", height,
                                    CFG.OCTA_RADIUS, step_z)
        col_mesh.materials.append(materials[name])

        for ix in range(CFG.GRID_X):
            for iy in range(CFG.GRID_Y):
                loc = (x0 + ix * step_xy, y0 + iy * step_xy, z_cursor)
                obj = new_stack_object(f"{name}_stack_{ix}_{iy}",
                                       col_mesh, coll, loc)
                objects[name].append(obj)

        layer_top = z_cursor + (height - 1) * step_z + 2 * CFG.OCTA_RADIUS
        bounds[name] = (z_cursor, layer_top)
        z_cursor = layer_top + CFG.LAYER_SEPARATION_Z

    return {
        "objects": objects,
        "bounds": bounds,
        "step_xy": step_xy,
        "lateral_extent": max(CFG.GRID_X, CFG.GRID_Y) * step_xy,
    }


# =============================================================================
# Animation helpers
# =============================================================================

def _iter_fcurves(obj):
    """Yield an object's F-Curves across Blender versions.

    Pre-4.4 actions expose `action.fcurves` directly. Blender 4.4+/5.x use
    'slotted actions' where F-Curves live in layers -> strips -> channelbags,
    and `action.fcurves` no longer exists.
    """
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    action = ad.action

    legacy = getattr(action, "fcurves", None)   # None on 5.x (attr removed)
    if legacy is not None:
        for fc in legacy:
            yield fc
        return

    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for cbag in getattr(strip, "channelbags", []):
                for fc in cbag.fcurves:
                    yield fc


def _set_interpolation(obj, interp="SINE", easing="EASE_IN_OUT"):
    for fc in _iter_fcurves(obj):
        for kp in fc.keyframe_points:
            kp.interpolation = interp
            kp.easing = easing


# =============================================================================
# Animation: sacrificial erosion (edge -> center)
# =============================================================================

def animate_erosion(sacrificial_objs, rng):
    if not sacrificial_objs:
        return

    # Distance of each stack from the layer center (XY), for edge->center order.
    dists = [Vector((o.location.x, o.location.y, 0.0)).length
             for o in sacrificial_objs]
    max_dist = max(dists) or 1.0

    span = max(1, CFG.EROSION_END - CFG.EROSION_START
               - CFG.EROSION_PIECE_DURATION)

    for obj, dist in zip(sacrificial_objs, dists):
        base_loc = obj.location.copy()
        base_scale = obj.scale.copy()

        # Outward direction in XY only (stream sideways, never down).
        radial = Vector((base_loc.x, base_loc.y, 0.0))
        if radial.length < 1e-4:
            ang = rng.uniform(0, 2 * math.pi)
            radial = Vector((math.cos(ang), math.sin(ang), 0.0))
        radial.normalize()

        # Edge stacks (large dist) start early; center stacks (small) start late.
        dist_norm = dist / max_dist
        delay = (1.0 - dist_norm) * span
        start = int(CFG.EROSION_START + delay
                    + rng.uniform(-CFG.EROSION_JITTER, CFG.EROSION_JITTER))
        start = max(CFG.EROSION_START, start)
        end = start + CFG.EROSION_PIECE_DURATION

        target = base_loc + radial * CFG.EROSION_STREAM_DISTANCE
        target.z += CFG.EROSION_STREAM_LIFT * rng.uniform(0.5, 1.5)

        # Location
        obj.location = base_loc
        obj.keyframe_insert("location", frame=start)
        obj.location = target
        obj.keyframe_insert("location", frame=end)

        # Scale -> ~0 (dissipate)
        obj.scale = base_scale
        obj.keyframe_insert("scale", frame=start)
        obj.scale = base_scale * 0.01
        obj.keyframe_insert("scale", frame=end)

        # Tumble
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.keyframe_insert("rotation_euler", frame=start)
        spin = CFG.EROSION_SPIN * rng.uniform(0.6, 1.4)
        obj.rotation_euler = (spin * rng.uniform(-1, 1),
                              spin * rng.uniform(-1, 1),
                              spin * rng.uniform(-1, 1))
        obj.keyframe_insert("rotation_euler", frame=end)

        # Hide once dissipated
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert("hide_viewport", frame=end - 1)
        obj.keyframe_insert("hide_render", frame=end - 1)
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert("hide_viewport", frame=end)
        obj.keyframe_insert("hide_render", frame=end)

        _set_interpolation(obj, "SINE", "EASE_OUT")


# =============================================================================
# Animation: water molecules
# =============================================================================

def build_and_animate_water(water_mesh, water_mat, geom, rng):
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
        gone = int(rng.uniform(CFG.EROSION_END, CFG.WATER_DISSIPATE_END))

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
        "substrate":   make_material("Substrate", CFG.COLOR_SUBSTRATE),
        "sacrificial": make_material("Sacrificial", CFG.COLOR_SACRIFICIAL),
        "film":        make_material("Film", CFG.COLOR_FILM),
    }
    water_mat = make_material("Water", CFG.COLOR_WATER,
                              alpha=CFG.WATER_ALPHA, emission_boost=0.3)
    water_mesh = make_uv_sphere_mesh("WaterMolecule", CFG.WATER_RADIUS,
                                     CFG.WATER_UV_SEGMENTS)

    geom = build_layers(materials)
    animate_erosion(geom["objects"]["sacrificial"], rng)
    build_and_animate_water(water_mesh, water_mat, geom, rng)

    setup_camera_and_lights(geom)
    setup_render_and_timeline()

    bpy.context.scene.frame_set(CFG.FRAME_START)

    total_stacks = sum(len(v) for v in geom["objects"].values())
    print("[membrane_liftoff] Built {} stacks + {} water molecules. "
          "Frames {}-{} @ {}fps."
          .format(total_stacks, CFG.WATER_COUNT,
                  CFG.FRAME_START, CFG.FRAME_END, CFG.FPS))


if __name__ == "__main__":
    main()
