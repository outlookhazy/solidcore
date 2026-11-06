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
