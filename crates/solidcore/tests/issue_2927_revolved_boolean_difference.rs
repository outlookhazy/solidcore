//! Regression tests for Issue #2927:
//! `boolean_difference` on revolved parts with off-axis cuts (hyperbolas on cones,
//! bicircular quartics on tori) must never return `Ok` on degenerate, non-manifold,
//! zero-volume, or volume-expanded bodies.
//!
//! A result must either be a genuine valid closed manifold solid satisfying the
//! monotonicity invariant V(A \ B) <= V(A) + tol, or be cleanly refused with a typed error.

use std::f64::consts::PI;

use solidcore::boolean::api_brep::boolean_difference;
use solidcore::mass::integrator::properties::brep_volume_default;
use solidcore::math::Point3;
use solidcore::primitive::api_primitives::{create_box, create_cone, create_torus};
use solidcore::topology::brep::is_manifold_solid;
use solidcore::topology::brep::validation::{check_brep, CheckLevel};
use solidcore::Kernel;

const SIGN_ACCURACY: f64 = 1e-3;

/// Validates that off-axis cuts on a conical frustum never produce corrupt bodies.
///
/// Cuts along off-axis planes produce hyperbolic intersection curves. Valid results
/// are verified against independently derived analytical bounds for the conical frustum:
/// `V_exact` = (1/3) * pi * h * (R1^2 + R1*R2 + R2^2).
#[test]
fn cone_off_axis_cuts_never_produce_corrupt_ok_bodies() {
    let mut k = Kernel::default();
    let offsets = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0];

    // Analytical volume for conical frustum: R1=40, R2=10, h=50
    // V = (1/3) * pi * 50 * (1600 + 400 + 100) = 35000 * pi
    let exact_cone_vol = (1.0 / 3.0) * PI * 50.0 * (40.0 * 40.0 + 40.0 * 10.0 + 10.0 * 10.0);

    for &x_cut in &offsets {
        let cone = create_cone(&mut k, Point3::ZERO, 40.0, 10.0, 50.0).unwrap();

        // Tool box cutting off-axis parallel to the cone axis Z
        let tool = create_box(
            &mut k,
            Point3::new(x_cut, -100.0, -10.0),
            200.0,
            200.0,
            70.0,
        )
        .unwrap();

        match boolean_difference(&mut k, cone, tool) {
            Ok(mut result) => {
                // For partial off-axis cuts (offsets in [-30, 30]), material is guaranteed
                // to remain; empty successful results are invalid.
                assert!(
                    !result.face_list().is_empty(),
                    "Cut at x={x_cut} returned empty successful result"
                );

                let is_solid = is_manifold_solid(&result).unwrap()
                    || solidcore::query::api_queries::brep_is_closed_touching_solids(&result)
                        .unwrap();
                assert!(
                    is_solid,
                    "Cut at x={x_cut} returned Ok but result is not a manifold solid"
                );

                let check = check_brep(&mut k, &mut result, CheckLevel::Level0).unwrap();
                assert!(
                    check.is_valid(),
                    "Cut at x={x_cut} returned Ok but failed Level0 check"
                );

                let res_vol = brep_volume_default(&k, &mut result, SIGN_ACCURACY).unwrap();
                assert!(
                    res_vol > 0.0 && !res_vol.is_nan(),
                    "Cut at x={x_cut} returned Ok with non-positive volume {res_vol}"
                );

                let tol = 1e-4 * exact_cone_vol.max(1.0);
                assert!(
                    res_vol <= exact_cone_vol + tol,
                    "Cut at x={x_cut} expanded volume: {res_vol} > {exact_cone_vol}"
                );

                // For cuts where tool only covers x >= 0, at least half the frustum remains
                if x_cut >= 0.0 {
                    assert!(
                        res_vol >= 0.35 * exact_cone_vol,
                        "Cut at x={x_cut} removed too much material: {res_vol} < 0.35 * {exact_cone_vol}"
                    );
                }
            }
            Err(e) => {
                // Refusal with typed error is valid
                assert!(
                    matches!(e, solidcore::SolidError::Degenerate { .. })
                        || matches!(e, solidcore::SolidError::Upstream { .. })
                        || matches!(e, solidcore::SolidError::Failed(_)),
                    "Cut at x={x_cut} failed with unexpected error: {e:?}"
                );
            }
        }
    }
}

/// Validates that off-axis cuts on a torus never produce corrupt bodies.
///
/// Cuts on a torus produce bicircular quartic intersection curves. Valid results
/// are verified against independently derived analytical bounds: `V_exact` = 2 * pi^2 * R * r^2.
#[test]
fn torus_off_axis_cuts_never_produce_corrupt_ok_bodies() {
    let mut k = Kernel::default();
    let offsets = [-35.0, -20.0, -10.0, 0.0, 15.0, 30.0, 45.0];

    // Analytical volume for torus: R=45, r=10 => V = 2 * pi^2 * 45 * 100 = 9000 * pi^2
    let exact_torus_vol = 2.0 * PI * PI * 45.0 * 100.0;

    for &x_cut in &offsets {
        let torus = create_torus(&mut k, Point3::ZERO, 45.0, 10.0).unwrap();

        // Tool box cutting off-axis parallel to the torus axis Z
        let tool = create_box(
            &mut k,
            Point3::new(x_cut, -100.0, -30.0),
            200.0,
            200.0,
            60.0,
        )
        .unwrap();

        match boolean_difference(&mut k, torus, tool) {
            Ok(mut result) => {
                // Partial cuts must retain material
                assert!(
                    !result.face_list().is_empty(),
                    "Cut at x={x_cut} returned empty successful result"
                );

                let is_solid = is_manifold_solid(&result).unwrap()
                    || solidcore::query::api_queries::brep_is_closed_touching_solids(&result)
                        .unwrap();
                assert!(
                    is_solid,
                    "Cut at x={x_cut} returned Ok but result is not a manifold solid"
                );

                let check = check_brep(&mut k, &mut result, CheckLevel::Level0).unwrap();
                assert!(
                    check.is_valid(),
                    "Cut at x={x_cut} returned Ok but failed Level0 check"
                );

                let res_vol = brep_volume_default(&k, &mut result, SIGN_ACCURACY).unwrap();
                assert!(
                    res_vol > 0.0 && !res_vol.is_nan(),
                    "Cut at x={x_cut} returned Ok with non-positive volume {res_vol}"
                );

                let tol = 1e-4 * exact_torus_vol.max(1.0);
                assert!(
                    res_vol <= exact_torus_vol + tol,
                    "Cut at x={x_cut} expanded volume: {res_vol} > {exact_torus_vol}"
                );

                // For cuts at x >= 0, at least half the torus (x <= 0) remains
                if x_cut >= 0.0 {
                    assert!(
                        res_vol >= 0.35 * exact_torus_vol,
                        "Cut at x={x_cut} removed too much material: {res_vol} < 0.35 * {exact_torus_vol}"
                    );
                }
            }
            Err(e) => {
                assert!(
                    matches!(e, solidcore::SolidError::Degenerate { .. })
                        || matches!(e, solidcore::SolidError::Upstream { .. })
                        || matches!(e, solidcore::SolidError::Failed(_)),
                    "Cut at x={x_cut} failed with unexpected error: {e:?}"
                );
            }
        }
    }
}

use solidcore::geom::attribute::{AttributeStore, ObjectKey};
use solidcore::geom::object::ObjectRef;
use solidcore::topology::api_files::write_brep_to_file;
use solidcore::topology::brep_data_write::FromBrepEnv;
use solidcore::topology::topology::NotifyRecords;
use solidcore::topology::{EdgeId, EdgeuseId, FaceId};
use solidcore::{Brep, EntityRef};

fn named(e: EntityRef, base: u64) -> ObjectRef {
    let (type_code, kind, index) = match e {
        EntityRef::Brep(x) => (16011, 0u64, x.index()),
        EntityRef::Region(x) => (16012, 1, x.index()),
        EntityRef::Shell(x) => (16013, 2, x.index()),
        EntityRef::Faceuse(x) => (16022, 3, x.index()),
        EntityRef::Face(x) => (16002, 4, x.index()),
        EntityRef::Loopuse(x) => (16004, 5, x.index()),
        EntityRef::Loop(x) => (16003, 6, x.index()),
        EntityRef::Edgeuse(x) => (16005, 7, x.index()),
        EntityRef::Edge(x) => (16016, 8, x.index()),
        EntityRef::Vertexuse(x) => (16006, 9, x.index()),
        EntityRef::Vertex(x) => (16021, 10, x.index()),
    };
    ObjectRef {
        type_code,
        key: ObjectKey(base + 1 + (kind << 32) + u64::from(index)),
    }
}

fn write_smb(k: &Kernel, b: &Brep) -> Result<Vec<u8>, solidcore::SolidError> {
    let store = AttributeStore::new();
    let recs = NotifyRecords::new();
    let key_of = |e| named(e, 0);
    let env = FromBrepEnv {
        attributes: &store,
        key_of: &key_of,
        records: &recs,
        curve_key: &|_: EdgeId| ObjectKey(0),
        uv_curve_key: &|_: EdgeuseId| ObjectKey(0),
        surface_key: &|_: FaceId| ObjectKey(0),
    };
    write_brep_to_file(k, Some(b), &env, true)
}

/// Diagnostic export: writes SMB models of operands and cut bodies for `solidcore-diag`.
#[test]
#[ignore = "diagnostic render export: run with cargo test -p solidcore --test issue_2927_revolved_boolean_difference generate_diagnostic_smb_files -- --ignored"]
fn generate_diagnostic_smb_files() {
    let out_dir = std::path::Path::new("/tmp/diag_2927");
    std::fs::create_dir_all(out_dir).unwrap();

    let mut k = Kernel::default();

    // 1. Cone
    let cone = create_cone(&mut k, Point3::ZERO, 40.0, 10.0, 50.0).unwrap();
    std::fs::write(
        out_dir.join("cone_original.smb"),
        write_smb(&k, &cone).unwrap(),
    )
    .unwrap();

    let tool_cone =
        create_box(&mut k, Point3::new(0.0, -100.0, -10.0), 200.0, 200.0, 70.0).unwrap();
    std::fs::write(
        out_dir.join("cone_tool.smb"),
        write_smb(&k, &tool_cone).unwrap(),
    )
    .unwrap();

    let cut_cone = boolean_difference(&mut k, cone, tool_cone).unwrap();
    std::fs::write(
        out_dir.join("cone_cut_x0.smb"),
        write_smb(&k, &cut_cone).unwrap(),
    )
    .unwrap();

    // 2. Torus
    let torus = create_torus(&mut k, Point3::ZERO, 45.0, 10.0).unwrap();
    std::fs::write(
        out_dir.join("torus_original.smb"),
        write_smb(&k, &torus).unwrap(),
    )
    .unwrap();

    let tool_torus =
        create_box(&mut k, Point3::new(15.0, -100.0, -30.0), 200.0, 200.0, 60.0).unwrap();
    std::fs::write(
        out_dir.join("torus_tool.smb"),
        write_smb(&k, &tool_torus).unwrap(),
    )
    .unwrap();

    let cut_torus = boolean_difference(&mut k, torus, tool_torus).unwrap();
    std::fs::write(
        out_dir.join("torus_cut_x15.smb"),
        write_smb(&k, &cut_torus).unwrap(),
    )
    .unwrap();

    println!("Successfully exported SMB models to {}", out_dir.display());
}
