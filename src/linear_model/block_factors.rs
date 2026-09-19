use super::Jacobian;
use crate::{EvaluationError, Real, SolverError, optimization::norm_inf};
use faer::{
    MatMut, MatRef, Par,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{cholesky::llt, triangular_solve},
};

#[derive(Debug)]
struct VariableBlock {
    col: usize,
    width: usize,
    start: usize,
    fallback: bool,
}

/// Normalized local Gram matrices and their reusable Cholesky factors.
pub(crate) struct BlockFactors<R> {
    blocks: Vec<VariableBlock>,
    owners: Vec<usize>,
    gram: Vec<R>,
    factors: Vec<R>,
    scratch: Option<MemBuffer>,
}

impl<R: std::fmt::Debug> std::fmt::Debug for BlockFactors<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockFactors")
            .field("blocks", &self.blocks)
            .finish_non_exhaustive()
    }
}

impl<R: Real> Default for BlockFactors<R> {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            owners: Vec::new(),
            gram: Vec::new(),
            factors: Vec::new(),
            scratch: None,
        }
    }
}

impl<R: Real> BlockFactors<R> {
    pub fn prepare(&mut self, jacobian: &Jacobian<R>, scales: &[R]) -> Result<(), SolverError> {
        self.blocks.clear();
        self.gram.clear();
        self.owners.resize(jacobian.cols, usize::MAX);
        self.owners.fill(usize::MAX);
        for b in &jacobian.blocks {
            if b.cols == 0 {
                continue;
            }
            let end = b
                .col
                .checked_add(b.cols)
                .filter(|&end| end <= jacobian.cols)
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

    /// Apply L^-T, or L^-1 when transposed. Failed blocks remain identity,
    /// suitable for preconditioning but not for exact elimination.
    pub fn apply_factor_inverse(&self, mut rhs: MatMut<'_, R>, transpose: bool) {
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
    }

    /// Read-only lower-triangular Gram views and their coordinate offsets.
    pub fn gram_blocks(&self) -> impl Iterator<Item = (usize, MatRef<'_, R>)> {
        self.blocks.iter().map(|b| {
            (
                b.col,
                MatRef::from_column_major_slice(
                    &self.gram[b.start..b.start + b.width * b.width],
                    b.width,
                    b.width,
                ),
            )
        })
    }

    /// Apply the inverse damped normal blocks within a coordinate range. The
    /// caller must reject failed factors; preconditioning fallback is not exact.
    pub fn solve_normal(&self, mut rhs: MatMut<'_, R>, offset: usize, lambda: R) {
        for b in &self.blocks {
            if b.col < offset || b.col >= offset + rhs.nrows() {
                continue;
            }
            debug_assert!(!b.fallback);
            let l = MatRef::from_column_major_slice(
                &self.factors[b.start..b.start + b.width * b.width],
                b.width,
                b.width,
            );
            let mut part = rhs.as_mut().subrows_mut(b.col - offset, b.width);
            triangular_solve::solve_lower_triangular_in_place(l, part.as_mut(), Par::Seq);
            triangular_solve::solve_upper_triangular_in_place(l.transpose(), part, Par::Seq);
        }
        for i in 0..rhs.nrows() {
            if self.owners[offset + i] == usize::MAX {
                for k in 0..rhs.ncols() {
                    // Unobserved coordinates have no RHS/couplings. At lambda=0,
                    // preserve their zero minimum-norm increment.
                    if lambda != R::zero() {
                        rhs[(i, k)] /= lambda;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linear_model::Block;
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
        let mut p = BlockFactors::default();
        p.prepare(&j, &[1., 1., 1.]).unwrap();
        assert_eq!(p.factor(0.), 1);
        let mut v = [2., 3., 4.];
        p.apply_factor_inverse(MatMut::from_column_major_slice_mut(&mut v, 3, 1), false);
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
        assert!(p.prepare(&j, &[1., 1., 1.]).is_err());
        j.blocks.clear();
        p.prepare(&j, &[1., 1., 1.]).unwrap();
        assert_eq!(p.factor(0.1), 0);
    }
}
