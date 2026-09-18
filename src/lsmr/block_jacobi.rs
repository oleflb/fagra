use super::*;
use faer::{
    linalg::{cholesky::llt, triangular_solve},
    matrix_free::{BiPrecond, Precond},
};
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

#[derive(Debug)]
struct VariableBlock {
    col: usize,
    width: usize,
    start: usize,
    fallback: bool,
}

/// Factors normalized local Gram blocks, then applies L^-T (or its transpose).
pub(super) struct BlockJacobi<R> {
    dimension: usize,
    enabled: bool,
    blocks: Vec<VariableBlock>,
    owners: Vec<usize>,
    gram: Vec<R>,
    factors: Vec<R>,
    scratch: Option<MemBuffer>,
    pub timed: bool,
    pub elapsed: Mutex<Duration>,
}

impl<R: std::fmt::Debug> std::fmt::Debug for BlockJacobi<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockJacobi")
            .field("blocks", &self.blocks)
            .finish_non_exhaustive()
    }
}

impl<R: Real> Default for BlockJacobi<R> {
    fn default() -> Self {
        Self {
            dimension: 0,
            enabled: false,
            blocks: Vec::new(),
            owners: Vec::new(),
            gram: Vec::new(),
            factors: Vec::new(),
            scratch: None,
            timed: false,
            elapsed: Mutex::new(Duration::ZERO),
        }
    }
}

impl<R: Real> BlockJacobi<R> {
    pub fn prepare(
        &mut self,
        jacobian: &Jacobian<R>,
        scales: &[R],
        enabled: bool,
    ) -> Result<(), SolverError> {
        self.dimension = jacobian.cols;
        self.enabled = enabled;
        self.blocks.clear();
        self.gram.clear();
        if !enabled {
            return Ok(());
        }
        self.owners.resize(self.dimension, usize::MAX);
        self.owners.fill(usize::MAX);
        for b in &jacobian.blocks {
            if b.cols == 0 {
                continue;
            }
            let end = b
                .col
                .checked_add(b.cols)
                .filter(|&end| end <= self.dimension)
                .ok_or(EvaluationError::DimensionMismatch)?;
            let owner = self.owners[b.col];
            let id = if owner == usize::MAX {
                if self.owners[b.col..end].iter().any(|&id| id != usize::MAX) {
                    return Err(EvaluationError::DimensionMismatch.into());
                }
                let id = self.blocks.len();
                let start = self.gram.len();
                let size = b
                    .cols
                    .checked_mul(b.cols)
                    .and_then(|n| start.checked_add(n))
                    .ok_or(EvaluationError::DimensionMismatch)?;
                self.gram.resize(size, R::zero());
                self.blocks.push(VariableBlock {
                    col: b.col,
                    width: b.cols,
                    start,
                    fallback: false,
                });
                self.owners[b.col..end].fill(id);
                id
            } else {
                if self.blocks[owner].col != b.col || self.blocks[owner].width != b.cols {
                    return Err(EvaluationError::DimensionMismatch.into());
                }
                owner
            };
            let start = self.blocks[id].start;
            for col in 0..b.cols {
                for row in col..b.cols {
                    let mut sum = R::zero();
                    for k in 0..b.rows {
                        sum += (jacobian.values[b.start + col * b.rows + k] / scales[b.col + col])
                            * (jacobian.values[b.start + row * b.rows + k] / scales[b.col + row]);
                    }
                    self.gram[start + col * b.cols + row] += sum;
                }
            }
        }
        self.factors.resize(self.gram.len(), R::zero());
        let width = self.blocks.iter().map(|b| b.width).max().unwrap_or(0);
        let req = llt::factor::cholesky_in_place_scratch::<R>(width, Par::Seq, Default::default());
        if !self
            .scratch
            .as_mut()
            .is_some_and(|s| MemStack::new(s).can_hold(req))
        {
            self.scratch = Some(MemBuffer::new(req));
        }
        Ok(())
    }

    pub fn factor(&mut self, lambda: R) -> usize {
        if !self.enabled {
            return 0;
        }
        self.factors.copy_from_slice(&self.gram);
        let mut fallbacks = 0;
        for b in &mut self.blocks {
            let values = &mut self.factors[b.start..b.start + b.width * b.width];
            for j in 0..b.width {
                values[j * b.width + j] += lambda;
            }
            b.fallback = norm_inf(values).is_err()
                || llt::factor::cholesky_in_place(
                    MatMut::from_column_major_slice_mut(values, b.width, b.width),
                    Default::default(),
                    Par::Seq,
                    MemStack::new(self.scratch.as_mut().unwrap()),
                    Default::default(),
                )
                .is_err();
            if !b.fallback {
                b.fallback = (0..b.width).any(|j| {
                    !values[j * b.width + j].is_finite() || values[j * b.width + j] <= R::zero()
                });
            }
            fallbacks += usize::from(b.fallback);
        }
        fallbacks
    }

    pub fn transform(&self, mut rhs: MatMut<'_, R>, transpose: bool) {
        if !self.enabled {
            return;
        }
        let start = self.timed.then(Instant::now);
        for b in &self.blocks {
            // Identity in normalized coordinates = the existing diagonal preconditioner.
            if b.fallback {
                continue;
            }
            let l = MatRef::from_column_major_slice(
                &self.factors[b.start..b.start + b.width * b.width],
                b.width,
                b.width,
            );
            let part = rhs.as_mut().subrows_mut(b.col, b.width);
            if transpose {
                triangular_solve::solve_lower_triangular_in_place(l, part, Par::Seq);
            } else {
                triangular_solve::solve_upper_triangular_in_place(l.transpose(), part, Par::Seq);
            }
        }
        if let Some(start) = start {
            *self.elapsed.lock().unwrap() += start.elapsed();
        }
    }
}

impl<R: Real> LinOp<R> for BlockJacobi<R> {
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
impl<R: Real> BiLinOp<R> for BlockJacobi<R> {
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
impl<R: Real> Precond<R> for BlockJacobi<R> {
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
impl<R: Real> BiPrecond<R> for BlockJacobi<R> {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn singular_and_nonfinite_blocks_fall_back_and_overlaps_are_rejected() {
        let mut j = Jacobian {
            rows: 1,
            cols: 3,
            blocks: vec![Block {
                row: 0,
                col: 0,
                rows: 1,
                cols: 2,
                start: 0,
            }],
            values: vec![1., 1.],
        };
        let mut p = BlockJacobi::default();
        p.prepare(&j, &[1., 1., 1.], true).unwrap();
        assert_eq!(p.factor(0.), 1);
        let mut v = [2., 3., 4.];
        p.transform(MatMut::from_column_major_slice_mut(&mut v, 3, 1), false);
        assert_eq!(v, [2., 3., 4.]);
        assert_eq!(p.factor(0.01), 0);
        p.gram[0] = f64::INFINITY;
        assert_eq!(p.factor(0.01), 1);
        j.blocks.push(Block {
            row: 0,
            col: 1,
            rows: 1,
            cols: 2,
            start: 0,
        });
        assert!(p.prepare(&j, &[1., 1., 1.], true).is_err());
        j.blocks.clear();
        p.prepare(&j, &[1., 1., 1.], true).unwrap();
        assert_eq!(p.factor(0.1), 0);
    }
}
