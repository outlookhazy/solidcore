//! Independent adversarial stress testing for Issue #2927:
//! Revolved part boolean difference, union, and intersection validation.
//!
//! Evaluates:
//! 1. Touching solids (isolated vertex, edge, and face contacts).
//! 2. Pathological inputs:
//!    - Cutting plane coinciding with apex of cone.
//!    - Cutting plane passing through revolution axis of cone and torus.
//!    - Cutting planes tangent to inner seam/throat, outer equator, and top crown of torus.
//!    - Near-grazing cuts across quadric and quartic surfaces.
//! 3. Volume bounds and monotonicity:
//!    - V(A \ B) <= V(A) + tol (no volume inflation / leakage).
//!    - Positive finite volume for all non-empty results.
//!    - Disjoint / inclusion-exclusion conservation.
//! 4. Topological contract:
//!    - All Ok outcomes must be valid manifold or closed-touching solids.
//!    - Failures must produce clean typed `SolidError` refusals, never panics.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use solidcore::boolean::api_brep::{boolean_difference, boolean_intersection, boolean_union};
use solidcore::mass::integrator::properties::brep_volume_default;
use solidcore::math::Point3;
use solidcore::primitive::api_primitives::{
    create_box, create_cone, create_cylinder, create_torus,
};
use solidcore::query::api_queries::brep_is_closed_touching_solids;
use solidcore::topology::brep::is_manifold_solid;
use solidcore::topology::brep::validation::{check_brep, CheckLevel};
use solidcore::{Brep, Kernel, SolidError};

const SIGN_ACCURACY: f64 = 1e-3;

/// Validates that a non-empty result satisfies the solid contract:
/// manifold or closed-touching solid topology, passes Level 0 validity,
/// and has finite positive volume.
fn assert_valid_solid_contract(k: &mut Kernel, b: &mut Brep, desc: &str) -> f64 {
    if b.face_list().is_empty() {
        return 0.0;
    }
    let solid = is_manifold_solid(b).unwrap() || brep_is_closed_touching_solids(b).unwrap();
    assert!(solid, "{desc}: result with faces is not a solid");

    let chk = check_brep(k, b, CheckLevel::Level0).unwrap();
    assert!(
        chk.is_valid(),
        "{desc}: failed Level0 check: {:?}",
        chk.lines()
    );

    let vol = brep_volume_default(k, b, SIGN_ACCURACY).unwrap();
    assert!(
        vol > 0.0 && !vol.is_nan(),
        "{desc}: volume is non-positive or NaN: {vol}"
    );
    vol
}

/// Asserts that an error returned by boolean operations is a recognized typed refusal.
fn assert_typed_refusal(err: &SolidError, desc: &str) {
    match err {
        SolidError::Degenerate { .. }
        | SolidError::Upstream { .. }
        | SolidError::Failed(_)
        | SolidError::InvalidInput { .. } => {}
        other => panic!("{desc}: unexpected error type: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 1. Touching solids & Union/Intersection/Difference consistency
// ---------------------------------------------------------------------------

/// Verifies boolean behavior when solids touch at an isolated vertex.
#[test]
fn touching_solids_vertex_contact() {
    let mut k = Kernel::default();
    // Box A: [0, 10] x [0, 10] x [0, 10]
    // Box B: [10, 20] x [10, 20] x [10, 20] (touches only at (10, 10, 10))
    let a = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let b = create_box(&mut k, Point3::new(10.0, 10.0, 10.0), 10.0, 10.0, 10.0).unwrap();

    let mut a_copy = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let vol_a = brep_volume_default(&k, &mut a_copy, SIGN_ACCURACY).unwrap();

    // Difference: disjoint interiors, A \ B should equal A
    match boolean_difference(&mut k, a, b) {
        Ok(mut res) => {
            let res_vol = assert_valid_solid_contract(&mut k, &mut res, "vertex contact diff");
            assert!(
                (res_vol - vol_a).abs() <= 1e-2 * vol_a,
                "vertex contact diff changed volume: {res_vol} vs {vol_a}"
            );
        }
        Err(e) => assert_typed_refusal(&e, "vertex contact diff"),
    }

    // Intersection: disjoint interiors, must produce an empty solid or typed refusal
    let a2 = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let b2 = create_box(&mut k, Point3::new(10.0, 10.0, 10.0), 10.0, 10.0, 10.0).unwrap();
    match boolean_intersection(&mut k, a2, b2) {
        Ok(res) => {
            assert!(
                res.face_list().is_empty(),
                "vertex contact intersection must produce an empty solid, got {} faces",
                res.face_list().len()
            );
        }
        Err(e) => assert_typed_refusal(&e, "vertex contact inter"),
    }

    // Union: touching at vertex produces closed touching solids
    let a3 = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let b3 = create_box(&mut k, Point3::new(10.0, 10.0, 10.0), 10.0, 10.0, 10.0).unwrap();
    match boolean_union(&mut k, a3, b3) {
        Ok(mut res) => {
            let u_vol = assert_valid_solid_contract(&mut k, &mut res, "vertex contact union");
            assert!(
                (u_vol - 2.0 * vol_a).abs() <= 1e-2 * (2.0 * vol_a),
                "vertex contact union volume mismatch: {u_vol} vs {}",
                2.0 * vol_a
            );
        }
        Err(e) => assert_typed_refusal(&e, "vertex contact union"),
    }
}

/// Verifies boolean behavior when solids touch along an isolated edge.
#[test]
fn touching_solids_edge_contact() {
    let mut k = Kernel::default();
    // Box A: [0, 10] x [0, 10] x [0, 10]
    // Box B: [10, 20] x [0, 10] x [10, 20] (touches along line x=10, z=10, y in [0, 10])
    let a = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let b = create_box(&mut k, Point3::new(10.0, 0.0, 10.0), 10.0, 10.0, 10.0).unwrap();

    let mut a_copy = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let vol_a = brep_volume_default(&k, &mut a_copy, SIGN_ACCURACY).unwrap();

    match boolean_difference(&mut k, a, b) {
        Ok(mut res) => {
            let res_vol = assert_valid_solid_contract(&mut k, &mut res, "edge contact diff");
            assert!(
                (res_vol - vol_a).abs() <= 1e-2 * vol_a,
                "edge contact diff volume mismatch: {res_vol} vs {vol_a}"
            );
        }
        Err(e) => assert_typed_refusal(&e, "edge contact diff"),
    }
}

/// Verifies boolean behavior when solids share an entire coplanar boundary face.
#[test]
fn touching_solids_face_contact() {
    let mut k = Kernel::default();
    // Box A: [0, 10] x [0, 10] x [0, 10]
    // Box B: [10, 20] x [0, 10] x [0, 10] (shares face at x=10)
    let a = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let b = create_box(&mut k, Point3::new(10.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();

    let mut a_copy = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let vol_a = brep_volume_default(&k, &mut a_copy, SIGN_ACCURACY).unwrap();

    // Difference: sharing face, disjoint interiors, A \ B should equal A
    match boolean_difference(&mut k, a, b) {
        Ok(mut res) => {
            let res_vol = assert_valid_solid_contract(&mut k, &mut res, "face contact diff");
            assert!(
                (res_vol - vol_a).abs() <= 1e-2 * vol_a,
                "face contact diff volume mismatch: {res_vol} vs {vol_a}"
            );
        }
        Err(e) => assert_typed_refusal(&e, "face contact diff"),
    }

    // Union: merges into single solid of volume 2 * vol_a
    let a2 = create_box(&mut k, Point3::new(0.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    let b2 = create_box(&mut k, Point3::new(10.0, 0.0, 0.0), 10.0, 10.0, 10.0).unwrap();
    match boolean_union(&mut k, a2, b2) {
        Ok(mut res) => {
            let u_vol = assert_valid_solid_contract(&mut k, &mut res, "face contact union");
            assert!(
                (u_vol - 2.0 * vol_a).abs() <= 1e-2 * (2.0 * vol_a),
                "face contact union volume: {u_vol} vs {}",
                2.0 * vol_a
            );
        }
        Err(e) => assert_typed_refusal(&e, "face contact union"),
    }
}

// ---------------------------------------------------------------------------
// 2. Pathological inputs: Cone apex & Torus seams
// ---------------------------------------------------------------------------

/// Verifies pathological cuts across the singular apex and revolution axis of a cone.
#[test]
fn cone_apex_pathological_cuts() {
    let mut k = Kernel::default();
    // Pointed cone with apex at (0, 0, 50), base at z=0, radius=40
    let make_cone = |k: &mut Kernel| create_cone(k, Point3::ZERO, 40.0, 0.0, 50.0).unwrap();

    let mut ref_cone = make_cone(&mut k);
    let orig_vol = brep_volume_default(&k, &mut ref_cone, SIGN_ACCURACY).unwrap();
    assert!(orig_vol > 0.0);

    // Case 1: Cutting plane exactly at apex z = 50 (tool above z=50)
    // Touches only the singular apex vertex
    let c1 = make_cone(&mut k);
    let t1 = create_box(
        &mut k,
        Point3::new(-100.0, -100.0, 50.0),
        200.0,
        200.0,
        50.0,
    )
    .unwrap();
    match boolean_difference(&mut k, c1, t1) {
        Ok(mut res) => {
            let vol = assert_valid_solid_contract(&mut k, &mut res, "cone apex horizontal cut");
            let tol = 1e-3 * orig_vol.max(1.0);
            assert!(
                (vol - orig_vol).abs() <= tol,
                "apex-contact difference must preserve full volume: {vol} vs {orig_vol}"
            );
        }
        Err(e) => assert_typed_refusal(&e, "cone apex horizontal cut"),
    }

    // Case 2: Cutting plane passing directly through apex and revolution axis (x = 0)
    // Slices cone into half-cone
    let c2 = make_cone(&mut k);
    let t2 = create_box(&mut k, Point3::new(0.0, -100.0, -10.0), 100.0, 200.0, 70.0).unwrap();
    match boolean_difference(&mut k, c2, t2) {
        Ok(mut res) => {
            let vol = assert_valid_solid_contract(&mut k, &mut res, "cone axial vertical cut");
            let tol = 1e-4 * orig_vol.max(1.0);
            assert!(vol <= orig_vol + tol, "volume leaked: {vol} > {orig_vol}");
            // Half-cone volume should be ~ 0.5 * orig_vol within measurement accuracy (1%)
            assert!(
                (vol - 0.5 * orig_vol).abs() <= 0.01 * orig_vol,
                "half-cone volume unexpected: {vol} vs expected {}",
                0.5 * orig_vol
            );
        }
        Err(e) => assert_typed_refusal(&e, "cone axial vertical cut"),
    }

    // Case 3: Infinitesimal offset from apex (x = 0.001)
    let c3 = make_cone(&mut k);
    let t3 = create_box(
        &mut k,
        Point3::new(0.001, -100.0, -10.0),
        100.0,
        200.0,
        70.0,
    )
    .unwrap();
    match boolean_difference(&mut k, c3, t3) {
        Ok(mut res) => {
            let vol = assert_valid_solid_contract(&mut k, &mut res, "cone near-apex cut");
            let tol = 1e-4 * orig_vol.max(1.0);
            assert!(vol <= orig_vol + tol, "volume leaked: {vol} > {orig_vol}");
        }
        Err(e) => assert_typed_refusal(&e, "cone near-apex cut"),
    }
}

/// Verifies pathological cuts on a torus across seams, equators, and throat boundaries.
#[test]
fn torus_pathological_cuts() {
    let mut k = Kernel::default();
    // Torus: major R = 45, minor r = 10
    // Inner throat is at radius R - r = 35
    // Outer equator is at radius R + r = 55
    // Top crown is at z = 10
    let make_torus = |k: &mut Kernel| create_torus(k, Point3::ZERO, 45.0, 10.0).unwrap();

    let mut ref_torus = make_torus(&mut k);
    let orig_vol = brep_volume_default(&k, &mut ref_torus, SIGN_ACCURACY).unwrap();
    assert!(orig_vol > 0.0);

    // Case 1: Cutting plane exactly at inner throat (x = 35.0)
    let tor1 = make_torus(&mut k);
    let t1 = create_box(&mut k, Point3::new(35.0, -100.0, -20.0), 100.0, 200.0, 40.0).unwrap();
    match boolean_difference(&mut k, tor1, t1) {
        Ok(mut res) => {
            let vol = assert_valid_solid_contract(&mut k, &mut res, "torus inner throat cut");
            let tol = 1e-4 * orig_vol.max(1.0);
            assert!(vol <= orig_vol + tol, "volume leaked: {vol} > {orig_vol}");
        }
        Err(e) => assert_typed_refusal(&e, "torus inner throat cut"),
    }

    // Case 2: Cutting plane passing through revolution axis (x = 0.0)
    let tor2 = make_torus(&mut k);
    let t2 = create_box(&mut k, Point3::new(0.0, -100.0, -20.0), 100.0, 200.0, 40.0).unwrap();
    match boolean_difference(&mut k, tor2, t2) {
        Ok(mut res) => {
            let vol = assert_valid_solid_contract(&mut k, &mut res, "torus axial cut");
            let tol = 1e-4 * orig_vol.max(1.0);
            assert!(vol <= orig_vol + tol, "volume leaked: {vol} > {orig_vol}");
            // Half torus should have volume ~ 0.5 * orig_vol within measurement accuracy (1%)
            assert!(
                (vol - 0.5 * orig_vol).abs() <= 0.01 * orig_vol,
                "half-torus volume unexpected: {vol} vs expected {}",
                0.5 * orig_vol
            );
        }
        Err(e) => assert_typed_refusal(&e, "torus axial cut"),
    }

    // Case 3: Cutting plane tangent to outer equator (x = 55.0)
    let tor3 = make_torus(&mut k);
    let t3 = create_box(&mut k, Point3::new(55.0, -100.0, -20.0), 100.0, 200.0, 40.0).unwrap();
    match boolean_difference(&mut k, tor3, t3) {
        Ok(mut res) => {
            let vol = assert_valid_solid_contract(&mut k, &mut res, "torus outer equator tangent");
            let tol = 5e-4 * orig_vol.max(1.0);
            assert!(
                (vol - orig_vol).abs() <= tol,
                "outer equator tangent cut removed material: {vol} vs {orig_vol}"
            );
        }
        Err(e) => assert_typed_refusal(&e, "torus outer equator tangent"),
    }

    // Case 4: Cutting plane tangent to top crown (z = 10.0)
    let tor4 = make_torus(&mut k);
    let t4 = create_box(
        &mut k,
        Point3::new(-100.0, -100.0, 10.0),
        200.0,
        200.0,
        40.0,
    )
    .unwrap();
    match boolean_difference(&mut k, tor4, t4) {
        Ok(mut res) => {
            let vol = assert_valid_solid_contract(&mut k, &mut res, "torus top crown tangent");
            let tol = 5e-4 * orig_vol.max(1.0);
            assert!(
                (vol - orig_vol).abs() <= tol,
                "top crown tangent cut removed material: {vol} vs {orig_vol}"
            );
        }
        Err(e) => assert_typed_refusal(&e, "torus top crown tangent"),
    }

    // Case 5: Cylinder tool placed exactly through center hole of torus (radius 30 < inner throat 35)
    // Disjoint interiors!
    let tor5 = make_torus(&mut k);
    let tool_cyl = create_cylinder(&mut k, Point3::new(0.0, 0.0, -30.0), 30.0, 60.0).unwrap();
    match boolean_difference(&mut k, tor5, tool_cyl) {
        Ok(mut res) => {
            let vol =
                assert_valid_solid_contract(&mut k, &mut res, "torus center hole cylinder diff");
            // Derivation: SIGN_ACCURACY is 1e-3, volume integration has bound ~ 0.05%
            let tol = 5e-4 * orig_vol.max(1.0);
            assert!(
                (vol - orig_vol).abs() <= tol,
                "disjoint tool removed material: {vol} vs {orig_vol}"
            );
        }
        Err(e) => assert_typed_refusal(&e, "torus center hole cylinder diff"),
    }
}

// ---------------------------------------------------------------------------
// 3. Dense Parameterized Sweep: Volume Monotonicity & Leakage Test
// ---------------------------------------------------------------------------

/// Sweeps cutting planes densely across a conical frustum, asserting monotonicity
/// V(A \ B) <= V(A) + tol for all cuts.
#[test]
fn dense_offset_sweep_monotonicity_cone() {
    let mut k = Kernel::default();
    let offsets = [
        -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0,
        15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
    ];

    let mut base_cone = create_cone(&mut k, Point3::ZERO, 40.0, 10.0, 50.0).unwrap();
    let orig_vol = brep_volume_default(&k, &mut base_cone, SIGN_ACCURACY).unwrap();

    for &x_cut in &offsets {
        let cone = create_cone(&mut k, Point3::ZERO, 40.0, 10.0, 50.0).unwrap();
        let tool = create_box(
            &mut k,
            Point3::new(x_cut, -100.0, -10.0),
            200.0,
            200.0,
            70.0,
        )
        .unwrap();

        match boolean_difference(&mut k, cone, tool) {
            Ok(mut res) => {
                if !res.face_list().is_empty() {
                    let vol = assert_valid_solid_contract(
                        &mut k,
                        &mut res,
                        &format!("cone cut x={x_cut}"),
                    );
                    let tol = 1e-4 * orig_vol.max(1.0);
                    assert!(
                        vol <= orig_vol + tol,
                        "Cut at x={x_cut} produced volume expansion: {vol} > {orig_vol}"
                    );
                }
            }
            Err(e) => assert_typed_refusal(&e, &format!("cone cut x={x_cut}")),
        }
    }
}
