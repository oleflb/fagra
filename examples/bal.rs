//! BAL benchmark: cargo run --release --example bal -- problem.txt [steps] [inner] [repeats] [scaled|block|identity|schur] [tolerance]
#[path = "bal/model.rs"]
mod model;
use fagra::{
    __private::DampedLeastSquaresBackend, LevenbergMarquardt, Lsmr, OptimizeOptions, Schur, Solver,
};
use model::*;
use std::{
    cell::Cell,
    error::Error,
    time::{Duration, Instant},
};

#[derive(Clone, Copy)]
struct Progress {
    start: Instant,
    initial: f64,
    reached: [Option<f64>; 2],
}
thread_local! {
    static PROGRESS: Cell<Option<Progress>> = const { Cell::new(None) };
}
fn progress(stats: fagra::LmStatistics) {
    PROGRESS.with(|p| {
        if let Some(mut progress) = p.get() {
            for (slot, fraction) in progress.reached.iter_mut().zip([0.05, 0.02]) {
                if slot.is_none() && stats.cost.is_some_and(|c| c <= progress.initial * fraction) {
                    *slot = Some(progress.start.elapsed().as_secs_f64());
                }
            }
            p.set(Some(progress));
        }
    });
}

fagra::states! { States { cameras: Camera, points: Point } }
fagra::factors! { Factors { frames: Batch<Frame, Reprojection> } }

struct Measurements {
    solves: usize,
    iterations: usize,
    preparation: Duration,
    factorization: Duration,
    linear_solve: Duration,
    products: Duration,
    preconditioner: Duration,
    verification: Duration,
}

trait BenchmarkBackend: DampedLeastSquaresBackend<Scalar = f64> {
    fn measurements(&self) -> Measurements;
    fn print_diagnostics(&self, run: usize);
}
impl BenchmarkBackend for Lsmr {
    fn measurements(&self) -> Measurements {
        let stats = self.statistics();
        let t = self.timings();
        Measurements {
            solves: stats.solves,
            iterations: stats.iterations,
            preparation: t.preparation,
            factorization: t.factorization,
            linear_solve: t.linear_solve,
            products: t.forward + t.transpose,
            preconditioner: t.preconditioner,
            verification: t.residual_checks,
        }
    }
    fn print_diagnostics(&self, run: usize) {
        eprintln!("run={run} last_linear={:?}", self.diagnostics());
    }
}
impl BenchmarkBackend for Schur {
    fn measurements(&self) -> Measurements {
        let stats = self.statistics();
        let t = self.timings();
        Measurements {
            solves: stats.solves,
            iterations: stats.iterations,
            preparation: t.preparation,
            factorization: t.factorization,
            linear_solve: t.linear_solve,
            products: t.products,
            preconditioner: t.preconditioner,
            verification: t.verification,
        }
    }
    fn print_diagnostics(&self, run: usize) {
        eprintln!("run={run} last_linear={:?}", self.diagnostics());
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: bal problem.txt [steps=20] [inner=1000] [repeats=3] [scaled|block|identity|schur] [tolerance=1e-6]")?;
    if args.len() > 7 {
        return Err("too many arguments".into());
    }
    let mode = args.get(5).map(String::as_str).unwrap_or("scaled");
    let (scaled, block) = match mode {
        "scaled" => (true, false),
        "block" => (true, true),
        "identity" => (false, false),
        "schur" => (true, true),
        _ => return Err("backend mode must be scaled, block, identity or schur".into()),
    };
    let tolerance: f64 = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(1e-6);
    if !tolerance.is_finite() || !(0.0..1.0).contains(&tolerance) {
        return Err("tolerance must be finite and in [0,1)".into());
    }
    let number = |i: usize, default: usize| -> Result<usize, Box<dyn Error>> {
        let n = args
            .get(i)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default);
        if n == 0 {
            return Err("limits and repeats must be positive".into());
        }
        Ok(n)
    };
    let steps = number(2, 20)?;
    let inner = number(3, 1000)?;
    let repeats = number(4, 3)?;
    let start = Instant::now();
    let problem = Problem::parse(&std::fs::read_to_string(path)?)?;
    let load = start.elapsed().as_secs_f64();
    eprintln!(
        "file={path} cameras={} points={} observations={} steps={steps} inner={inner} repeats={repeats} mode={mode} rel_tol={tolerance} gradient_tol=1e-8 max_trials=16 gauge=free threads=1",
        problem.cameras.len(),
        problem.points.len(),
        problem.observations.len()
    );
    println!(
        "run,load_s,build_s,solve_s,status,initial_cost,final_cost,rmse,last_gradient,accepted,rejected,linearizations,solves,inner_iterations,cost_evaluations,last_rejection,damping,peak_rss_kib,target_5pct_s,target_2pct_s,prepare_s,factor_s,linear_s,products_s,precondition_s,verification_s"
    );
    let trace = std::env::var_os("BAL_TRACE").is_some();
    if mode == "schur" {
        benchmark(&problem, load, steps, repeats, || {
            let mut backend = Schur::new(problem.cameras.len() * 9, 9, 3);
            backend.max_iterations = inner;
            backend.relative_tolerance = tolerance;
            backend.collect_timings = true;
            if trace {
                backend.on_solve = Some(|d| eprintln!("linear_attempt={d:?}"));
            }
            backend
        })
    } else {
        benchmark(&problem, load, steps, repeats, || {
            let mut backend = Lsmr::default();
            backend.max_iterations = inner;
            backend.relative_tolerance = tolerance;
            backend.diagonal_preconditioning = scaled;
            backend.block_preconditioning = block;
            backend.collect_timings = true;
            if trace {
                backend.on_solve = Some(|d| eprintln!("linear_attempt={d:?}"));
            }
            backend
        })
    }
}

fn benchmark<B: BenchmarkBackend>(
    problem: &Problem,
    load: f64,
    steps: usize,
    repeats: usize,
    backend: impl Fn() -> B,
) -> Result<(), Box<dyn Error>> {
    for run in 1..=repeats {
        let start = Instant::now();
        let mut graph = Solver::<States, Factors>::new();
        let cameras: Vec<_> = problem
            .cameras
            .iter()
            .cloned()
            .map(|c| graph.add(c))
            .collect();
        let points: Vec<_> = problem
            .points
            .iter()
            .cloned()
            .map(|p| graph.add(p))
            .collect();
        let frames: Vec<_> = cameras
            .iter()
            .map(|&camera| graph.add_batch(Frame { camera }))
            .collect();
        for o in &problem.observations {
            graph.add_factor_to(
                frames[o.camera],
                Reprojection {
                    point: points[o.point],
                    pixel: o.pixel,
                },
            )?;
        }
        let build = start.elapsed().as_secs_f64();
        // Independently evaluate the published BAL formula, without the factor implementation.
        let objective = |graph: &Solver<States, Factors>| -> Result<f64, Box<dyn Error>> {
            use faer_ext::nalgebra::{Rotation3, Vector3};
            let mut cost = 0.;
            for o in &problem.observations {
                let c = &graph.get(cameras[o.camera])?.0;
                let x = graph.get(points[o.point])?.0;
                let p = Rotation3::new(Vector3::new(c[0], c[1], c[2])) * x
                    + Vector3::new(c[3], c[4], c[5]);
                let u = -p.x / p.z;
                let v = -p.y / p.z;
                let r2 = u * u + v * v;
                let scale = c[6] * (1. + c[7] * r2 + c[8] * r2 * r2);
                cost += 0.5 * ((scale * u - o.pixel.x).powi(2) + (scale * v - o.pixel.y).powi(2));
            }
            Ok(cost)
        };
        let initial = objective(&graph)?;
        let mut lm = LevenbergMarquardt::new(backend());
        lm.on_accept = Some(progress);
        let options = OptimizeOptions {
            max_iterations: steps,
            gradient_tolerance: 1e-8,
            step_tolerance: 0.,
            cost_tolerance: 0.,
        };
        let start = Instant::now();
        PROGRESS.with(|p| {
            p.set(Some(Progress {
                start,
                initial,
                reached: [None, None],
            }))
        });
        let result = graph.optimize_with(&mut lm, &options);
        let elapsed = start.elapsed().as_secs_f64();
        let cost = objective(&graph)?;
        if !cost.is_finite()
            || (lm.statistics().cost.unwrap_or(initial) - cost).abs() > 1e-8 * cost.max(1.)
        {
            return Err("independent objective mismatch".into());
        }
        let stats = lm.statistics();
        let measurements = lm.backend().measurements();
        let targets = PROGRESS.with(|p| p.take().unwrap().reached);
        let target = |i: usize| {
            targets[i]
                .map(|t| format!("{t:.6}"))
                .unwrap_or_else(|| "NA".into())
        };
        lm.backend().print_diagnostics(run);
        let rss = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmHWM:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "NA".into());
        let status = match result {
            Ok(r) => format!("{:?}", r.termination),
            Err(e) => format!("{e:?}"),
        };
        println!(
            "{run},{load:.6},{build:.6},{elapsed:.6},{status},{initial:.12e},{cost:.12e},{rmse:.8},{gradient:.8e},{accepted},{rejected},{linearizations},{solves},{iterations},{costs},{last_rejection:?},{damping:.8e},{rss},{target5},{target2},{preparation:.6},{factorization:.6},{linear:.6},{products:.6},{preconditioner:.6},{verification:.6}",
            rmse = (2. * cost / problem.observations.len() as f64).sqrt(),
            gradient = stats.gradient_norm.unwrap_or(f64::NAN),
            accepted = stats.accepted_steps,
            rejected = stats.rejected_steps,
            linearizations = stats.linearizations,
            solves = measurements.solves,
            iterations = measurements.iterations,
            costs = stats.cost_evaluations,
            last_rejection = stats.last_rejection,
            damping = stats.damping,
            target5 = target(0),
            target2 = target(1),
            preparation = measurements.preparation.as_secs_f64(),
            factorization = measurements.factorization.as_secs_f64(),
            linear = measurements.linear_solve.as_secs_f64(),
            products = measurements.products.as_secs_f64(),
            preconditioner = measurements.preconditioner.as_secs_f64(),
            verification = measurements.verification.as_secs_f64()
        );
    }
    Ok(())
}
