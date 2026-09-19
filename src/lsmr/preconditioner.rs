//! LSMR's block-factor adapter; fallback and instrumentation stay solver-local.
use super::*;
use faer::matrix_free::Precond;

#[derive(Debug)]
pub(super) struct FactorPreconditioner<'a, R> {
    pub dimension: usize,
    pub factors: Option<&'a BlockFactors<R>>,
    pub timings: Option<&'a Mutex<LsmrTimings>>,
}
impl<R: Real> FactorPreconditioner<'_, R> {
    fn transform(&self, rhs: MatMut<'_, R>, transpose: bool) {
        if let Some(factors) = self.factors {
            let start = self.timings.map(|_| Instant::now());
            factors.apply_factor_inverse(rhs, transpose);
            if let (Some(clock), Some(start)) = (self.timings, start) {
                clock.lock().unwrap().preconditioner += start.elapsed();
            }
        }
    }
}
impl<R: Real> LinOp<R> for FactorPreconditioner<'_, R> {
    fn nrows(&self) -> usize {
        self.dimension
    }
    fn ncols(&self) -> usize {
        self.dimension
    }
    fn apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn apply(&self, mut out: MatMut<'_, R>, rhs: MatRef<'_, R>, _: Par, _: &mut MemStack) {
        out.copy_from(rhs);
        self.transform(out, false);
    }
    fn conj_apply(&self, out: MatMut<'_, R>, rhs: MatRef<'_, R>, par: Par, stack: &mut MemStack) {
        self.apply(out, rhs, par, stack);
    }
}
impl<R: Real> BiLinOp<R> for FactorPreconditioner<'_, R> {
    fn transpose_apply_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn transpose_apply(
        &self,
        mut out: MatMut<'_, R>,
        rhs: MatRef<'_, R>,
        _: Par,
        _: &mut MemStack,
    ) {
        out.copy_from(rhs);
        self.transform(out, true);
    }
    fn adjoint_apply(
        &self,
        out: MatMut<'_, R>,
        rhs: MatRef<'_, R>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.transpose_apply(out, rhs, par, stack);
    }
}
impl<R: Real> Precond<R> for FactorPreconditioner<'_, R> {
    fn apply_in_place_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn apply_in_place(&self, rhs: MatMut<'_, R>, _: Par, _: &mut MemStack) {
        self.transform(rhs, false);
    }
    fn conj_apply_in_place(&self, rhs: MatMut<'_, R>, par: Par, stack: &mut MemStack) {
        self.apply_in_place(rhs, par, stack);
    }
}
impl<R: Real> BiPrecond<R> for FactorPreconditioner<'_, R> {
    fn transpose_apply_in_place_scratch(&self, _: usize, _: Par) -> StackReq {
        StackReq::EMPTY
    }
    fn transpose_apply_in_place(&self, rhs: MatMut<'_, R>, _: Par, _: &mut MemStack) {
        self.transform(rhs, true);
    }
    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, R>, par: Par, stack: &mut MemStack) {
        self.transpose_apply_in_place(rhs, par, stack);
    }
}
