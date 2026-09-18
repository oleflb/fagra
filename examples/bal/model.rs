use faer_ext::nalgebra::{
    Const, DefaultAllocator, Matrix3, SMatrix, SVector, UnitQuaternion, Vector2, Vector3,
};
use fagra::{
    BlockId, EvaluationError, FactorBatch, FactorSelection, Jacobian, JacobianBlock,
    LinearizationSink, StateKey, StateStore, Tangent, Variable,
};

macro_rules! vector {
    ($name:ident, $n:literal) => {
        #[derive(Clone, Debug)]
        pub struct $name(pub SVector<f64, $n>);
        impl Variable for $name {
            type Scalar = f64;
            type Dim = Const<$n>;
            type Allocator = DefaultAllocator;
            fn identity() -> Self {
                Self(SVector::zeros())
            }
            fn compose(&self, other: &Self) -> Self {
                Self(self.0 + other.0)
            }
            fn inverse(&self) -> Self {
                Self(-self.0)
            }
            fn exp(d: &Tangent<Self>) -> Self {
                Self(*d)
            }
            fn log(&self) -> Tangent<Self> {
                self.0
            }
            fn adjoint(&self) -> Jacobian<Self> {
                SMatrix::identity()
            }
            fn right_jacobian(_: &Tangent<Self>) -> Jacobian<Self> {
                SMatrix::identity()
            }
            fn right_jacobian_inverse(_: &Tangent<Self>) -> Jacobian<Self> {
                SMatrix::identity()
            }
        }
    };
}
vector!(Camera, 9);
vector!(Point, 3);

pub struct Observation {
    pub camera: usize,
    pub point: usize,
    pub pixel: Vector2<f64>,
}
pub struct Problem {
    pub cameras: Vec<Camera>,
    pub points: Vec<Point>,
    pub observations: Vec<Observation>,
}

impl Problem {
    pub fn parse(text: &str) -> Result<Self, String> {
        fn next<T: std::str::FromStr>(it: &mut std::str::SplitWhitespace<'_>) -> Result<T, String> {
            it.next()
                .ok_or("truncated BAL input")?
                .parse()
                .map_err(|_| "invalid BAL token".into())
        }
        let mut it = text.split_whitespace();
        let nc: usize = next(&mut it)?;
        let np: usize = next(&mut it)?;
        let no: usize = next(&mut it)?;
        if nc == 0 || np == 0 || no == 0 {
            return Err("empty BAL problem".into());
        }
        let mut observations = Vec::new();
        for _ in 0..no {
            let camera = next(&mut it)?;
            let point = next(&mut it)?;
            let pixel = Vector2::new(next(&mut it)?, next(&mut it)?);
            if camera >= nc || point >= np || !pixel.iter().all(|x: &f64| x.is_finite()) {
                return Err("invalid observation index or value".into());
            }
            observations.push(Observation {
                camera,
                point,
                pixel,
            });
        }
        fn values<const N: usize>(
            it: &mut std::str::SplitWhitespace<'_>,
        ) -> Result<SVector<f64, N>, String> {
            let mut v = SVector::<f64, N>::zeros();
            for x in v.iter_mut() {
                *x = next(it)?;
                if !f64::is_finite(*x) {
                    return Err("nonfinite parameter".into());
                }
            }
            Ok(v)
        }
        let cameras = (0..nc)
            .map(|_| values(&mut it).map(Camera))
            .collect::<Result<_, _>>()?;
        let points = (0..np)
            .map(|_| values(&mut it).map(Point))
            .collect::<Result<_, _>>()?;
        if it.next().is_some() {
            return Err("trailing BAL tokens".into());
        }
        Ok(Self {
            cameras,
            points,
            observations,
        })
    }
}

fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0., -v.z, v.y, v.z, 0., -v.x, -v.y, v.x, 0.)
}

pub struct Projection {
    camera: SVector<f64, 9>,
    rotation: Matrix3<f64>,
    right: Matrix3<f64>,
}
impl Projection {
    pub fn new(camera: &Camera) -> Self {
        let w = camera.0.fixed_rows::<3>(0).into_owned();
        let t2 = w.norm_squared();
        let (a, b) = if t2 < 1e-8 {
            (
                0.5 - t2 / 24. + t2 * t2 / 720.,
                1. / 6. - t2 / 120. + t2 * t2 / 5040.,
            )
        } else {
            let t = t2.sqrt();
            ((1. - t.cos()) / t2, (t - t.sin()) / (t2 * t))
        };
        let k = skew(w);
        Self {
            camera: camera.0,
            rotation: UnitQuaternion::from_scaled_axis(w)
                .to_rotation_matrix()
                .into_inner(),
            right: Matrix3::identity() - a * k + b * k * k,
        }
    }
    pub fn value(&self, point: &Point) -> Result<Vector2<f64>, EvaluationError> {
        let p = self.rotation * point.0 + self.camera.fixed_rows::<3>(3);
        let q = -p.fixed_rows::<2>(0) / p.z;
        let r2 = q.norm_squared();
        let pixel = self.camera[6] * (1. + self.camera[7] * r2 + self.camera[8] * r2 * r2) * q;
        if p.z == 0. || !pixel.iter().all(|x| x.is_finite()) {
            return Err(EvaluationError::InvalidEvaluation);
        }
        Ok(pixel)
    }
    pub fn jacobians(&self, point: &Point) -> (SMatrix<f64, 2, 9>, SMatrix<f64, 2, 3>) {
        let p = self.rotation * point.0 + self.camera.fixed_rows::<3>(3);
        let q = -p.fixed_rows::<2>(0) / p.z;
        let r2 = q.norm_squared();
        let f = self.camera[6];
        let radial = 1. + self.camera[7] * r2 + self.camera[8] * r2 * r2;
        let dq = f
            * (radial * SMatrix::<f64, 2, 2>::identity()
                + (2. * self.camera[7] + 4. * self.camera[8] * r2) * q * q.transpose());
        let dp = dq
            * SMatrix::<f64, 2, 3>::new(
                -1. / p.z,
                0.,
                p.x / (p.z * p.z),
                0.,
                -1. / p.z,
                p.y / (p.z * p.z),
            );
        let mut camera = SMatrix::<f64, 2, 9>::zeros();
        camera
            .fixed_view_mut::<2, 3>(0, 0)
            .copy_from(&(dp * -self.rotation * skew(point.0) * self.right));
        camera.fixed_view_mut::<2, 3>(0, 3).copy_from(&dp);
        camera.column_mut(6).copy_from(&(radial * q));
        camera.column_mut(7).copy_from(&(f * r2 * q));
        camera.column_mut(8).copy_from(&(f * r2 * r2 * q));
        (camera, dp * self.rotation)
    }
}

pub struct Frame {
    pub camera: StateKey<Camera>,
}
pub struct Reprojection {
    pub point: StateKey<Point>,
    pub pixel: Vector2<f64>,
}
impl<S: StateStore<Camera> + StateStore<Point>> FactorBatch<S> for Frame {
    type Scalar = f64;
    type Factor = Reprojection;
    fn visit_variables(&self, factor: &Reprojection, mut visit: impl FnMut(BlockId)) {
        visit(self.camera.block_id());
        visit(factor.point.block_id());
    }
    fn cost(
        &self,
        states: &S,
        factors: FactorSelection<'_, Reprojection>,
    ) -> Result<f64, EvaluationError> {
        let projection = Projection::new(states.get(self.camera)?);
        let mut cost = 0.;
        for (_, factor) in factors {
            cost +=
                0.5 * (projection.value(states.get(factor.point)?)? - factor.pixel).norm_squared();
        }
        Ok(cost)
    }
    fn linearize<L: LinearizationSink<Scalar = f64>>(
        &self,
        states: &S,
        factors: FactorSelection<'_, Reprojection>,
        sink: &mut L,
    ) -> Result<(), EvaluationError> {
        let projection = Projection::new(states.get(self.camera)?);
        for (id, factor) in factors {
            let point = states.get(factor.point)?;
            let residual = projection.value(point)? - factor.pixel;
            let (jc, jp) = projection.jacobians(point);
            sink.factor(id, |out| {
                out.residual(
                    &residual,
                    &[
                        JacobianBlock::new(self.camera, &jc),
                        JacobianBlock::new(factor.point, &jp),
                    ],
                )
            })?;
        }
        Ok(())
    }
}
