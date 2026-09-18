//! Square-root elimination and solver-owned, fixed-reference manifold priors.

use std::{collections::HashSet, convert::Infallible, ops::Range};

use faer::{
    Conj, Mat, Par,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{householder, qr::col_pivoting::factor},
};
use faer_ext::{
    IntoNalgebra,
    nalgebra::{DMatrixView, DimName, Dyn},
};

use crate::{
    BlockId, EvaluationError, Factor, FactorBatch, FactorId, KeyError, LinearizationSink, Real,
    SolverError, StateKey, Variable,
    dense::DensePool,
    factors::{FactorSchema, FactorVisitor},
    key::RawKey,
    optimization::{Block, CheckedSink, Layout, LeastSquaresBackend},
    states::{StateSchema, StateVisitor},
    storage::{BatchPool, FactorPool, StatePool, checked_cost},
};

/// Numerical controls for square-root marginalization.
#[derive(Debug, Clone, Copy)]
pub struct MarginalizationOptions<R: Real = f64> {
    /// Relative cutoff for the pivoted QR diagonal of the eliminated columns.
    /// `None` uses machine epsilon times the larger matrix dimension. Values
    /// must be finite and in [0, 1). Directions below the cutoff are treated as
    /// zero in the eliminated block. This changes the reduced approximation;
    /// no damping or diagonal regularization is added.
    pub relative_rank_tolerance: Option<R>,
}

impl<R: Real> Default for MarginalizationOptions<R> {
    fn default() -> Self {
        Self {
            relative_rank_tolerance: None,
        }
    }
}

/// Dimensions and graph edits performed by a successful marginalization.
#[derive(Debug, Default, Clone, Copy)]
pub struct MarginalizationReport {
    /// Number of distinct removed state handles.
    pub removed_states: usize,
    /// Number of removed tangent coordinates.
    pub removed_dof: usize,
    /// Number of absorbed ordinary factors and batch payloads.
    pub absorbed_factors: usize,
    /// Number of absorbed internal marginal priors.
    pub absorbed_priors: usize,
    /// Number of surviving coordinates in the new prior's separator.
    pub separator_dof: usize,
    /// Numerical rank of the eliminated Jacobian columns.
    pub eliminated_rank: usize,
    /// Stored residual rows in the replacement prior.
    pub prior_rows: usize,
}

pub(crate) struct Anchor<T: Variable> {
    prior: RawKey,
    state: StateKey<T>,
    column: usize,
    inverse: T,
}

struct Prior<R> {
    blocks: Vec<(BlockId, Range<usize>)>,
    a: Mat<R>,
    b: Vec<R>,
    residual: Vec<R>,
    jacobian: Mat<R>,
}

impl<R: Real> Default for Prior<R> {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            a: Mat::new(),
            b: Vec::new(),
            residual: Vec::new(),
            jacobian: Mat::new(),
        }
    }
}

/// Internal prior storage passed to nonlinear optimizers.
#[doc(hidden)]
pub struct Priors<R: Real> {
    entries: DensePool<Prior<R>>,
    pub(crate) constant: R,
}

impl<R: Real> Default for Priors<R> {
    fn default() -> Self {
        Self {
            entries: DensePool::default(),
            constant: R::zero(),
        }
    }
}

impl<R: Real> Priors<R> {
    fn update<S: StateSchema<Scalar = R>>(
        &mut self,
        states: &mut S,
        derivatives: bool,
        selected: Option<&HashSet<FactorId>>,
    ) -> Result<(), SolverError> {
        for (local, prior) in self.entries.keys.iter().zip(&mut self.entries.values) {
            let id = FactorId(RawKey {
                pool: self.entries.id,
                local: *local,
            });
            if selected.is_none_or(|ids| ids.contains(&id)) {
                prior.residual.copy_from_slice(&prior.b);
            }
        }
        struct Evaluate<'a, R: Real> {
            priors: &'a mut Priors<R>,
            derivatives: bool,
            selected: Option<&'a HashSet<FactorId>>,
        }
        impl<R: Real> StateVisitor<R> for Evaluate<'_, R> {
            type Error = SolverError;
            fn pool<T: Variable<Scalar = R>>(
                &mut self,
                pool: &mut StatePool<T>,
            ) -> Result<(), SolverError> {
                for anchor in &pool.anchors {
                    if self
                        .selected
                        .is_some_and(|ids| !ids.contains(&FactorId(anchor.prior)))
                    {
                        continue;
                    }
                    let delta = anchor.inverse.compose(pool.get(anchor.state)?).log();
                    if delta.iter().any(|v| !v.is_finite()) {
                        return Err(EvaluationError::InvalidEvaluation.into());
                    }
                    let prior = self.priors.entries.get_mut(anchor.prior)?;
                    let start = anchor.column;
                    for row in 0..prior.a.nrows() {
                        for col in 0..T::Dim::DIM {
                            prior.residual[row] += prior.a[(row, start + col)] * delta[col];
                        }
                    }
                    if self.derivatives {
                        let chart = T::right_jacobian_inverse(&delta);
                        if chart.iter().any(|v| !v.is_finite()) {
                            return Err(EvaluationError::InvalidEvaluation.into());
                        }
                        for col in 0..T::Dim::DIM {
                            for row in 0..prior.a.nrows() {
                                let mut value = R::zero();
                                for k in 0..T::Dim::DIM {
                                    value += prior.a[(row, start + k)] * chart[(k, col)];
                                }
                                prior.jacobian[(row, start + col)] = value;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
        states.visit(&mut Evaluate {
            priors: self,
            derivatives,
            selected,
        })?;
        for (key, prior) in self.entries.iter() {
            if selected.is_none_or(|ids| ids.contains(&FactorId(key)))
                && (prior.residual.iter().any(|v| !v.is_finite())
                    || (derivatives && !finite(&prior.jacobian)))
            {
                return Err(EvaluationError::InvalidEvaluation.into());
            }
        }
        Ok(())
    }

    pub(crate) fn cost<S: StateSchema<Scalar = R>>(
        &mut self,
        states: &mut S,
    ) -> Result<R, SolverError> {
        self.update(states, false, None)?;
        checked_cost(Ok(self
            .entries
            .values
            .iter()
            .flat_map(|p| p.residual.iter())
            .fold(R::zero(), |sum, &r| sum + (R::from_f64_impl(0.5) * r) * r)))
    }

    pub(crate) fn linearize<S: StateSchema<Scalar = R>, B: LeastSquaresBackend<Scalar = R>>(
        &mut self,
        states: &mut S,
        backend: &mut B,
        layout: &Layout,
        selected: Option<&HashSet<FactorId>>,
    ) -> Result<(), SolverError> {
        self.update(states, true, selected)?;
        for (key, prior) in self.entries.iter() {
            if selected.is_some_and(|ids| !ids.contains(&FactorId(key))) {
                continue;
            }
            backend.accumulate(
                prior.residual.as_slice(),
                prior.blocks.iter().map(|(id, range)| {
                    let column = layout.blocks[layout.block_index[id]].offset;
                    (
                        column,
                        prior
                            .jacobian
                            .as_ref()
                            .subcols(range.start, range.len())
                            .into_nalgebra(),
                    )
                }),
            )?;
        }
        Ok(())
    }
}

#[derive(Default)]
struct Rows<R> {
    dimension: usize,
    values: Vec<R>, // Row-major [J | residual], packed once into the QR workspace.
}

impl<R: Real> LeastSquaresBackend for Rows<R> {
    type Scalar = R;
    fn prepare(&mut self, dimension: usize) -> Result<(), SolverError> {
        self.dimension = dimension;
        self.clear();
        Ok(())
    }
    fn clear(&mut self) {
        self.values.clear();
    }
    fn accumulate<'a>(
        &mut self,
        residual: &[R],
        jacobians: impl Iterator<Item = (usize, DMatrixView<'a, R, Dyn, Dyn>)> + Clone,
    ) -> Result<(), EvaluationError> {
        let stride = self
            .dimension
            .checked_add(1)
            .ok_or(EvaluationError::DimensionMismatch)?;
        let start = self.values.len();
        let end = residual
            .len()
            .checked_mul(stride)
            .and_then(|n| start.checked_add(n))
            .ok_or(EvaluationError::DimensionMismatch)?;
        self.values.resize(end, R::zero());
        for (row, &value) in residual.iter().enumerate() {
            self.values[start + row * stride + self.dimension] = value;
        }
        for (column, matrix) in jacobians {
            for col in 0..matrix.ncols() {
                for row in 0..residual.len() {
                    self.values[start + row * stride + column + col] = matrix[(row, col)];
                }
            }
        }
        Ok(())
    }
    fn gradient_norm(&self) -> Result<R, SolverError> {
        unreachable!("row collector")
    }
    fn solve(&mut self) -> Result<&[R], SolverError> {
        unreachable!("row collector")
    }
}

struct Elimination<R: Real> {
    rows: Rows<R>,
    matrix: Mat<R>,
    coefficients: Mat<R>,
    permutation: Vec<usize>,
    inverse: Vec<usize>,
    scratch: Option<MemBuffer>,
}

impl<R: Real> Default for Elimination<R> {
    fn default() -> Self {
        Self {
            rows: Rows {
                dimension: 0,
                values: Vec::new(),
            },
            matrix: Mat::new(),
            coefficients: Mat::new(),
            permutation: Vec::new(),
            inverse: Vec::new(),
            scratch: None,
        }
    }
}

fn finite<R: Real>(matrix: &Mat<R>) -> bool {
    (0..matrix.ncols()).all(|col| (0..matrix.nrows()).all(|row| matrix[(row, col)].is_finite()))
}

impl<R: Real> Elimination<R> {
    /// Factor only the specified columns, transforming all columns to their right.
    fn qr(&mut self, row: usize, column: usize, width: usize) {
        let height = self.matrix.nrows() - row;
        if height == 0 || width == 0 {
            return;
        }
        self.permutation.resize(width, 0);
        for (i, p) in self.permutation.iter_mut().enumerate() {
            *p = i;
        }
        // faer 0.24's pivoted QR normalizes by the largest column norm, even
        // when it is zero. A zero block needs neither pivoting nor reflectors.
        if (column..column + width)
            .all(|j| (row..self.matrix.nrows()).all(|i| self.matrix[(i, j)] == R::zero()))
        {
            return;
        }
        let block = 32.min(height.min(width));
        self.coefficients
            .resize_with(block, height.min(width), |_, _| R::zero());
        self.inverse.resize(width, 0);
        let trailing = self.matrix.ncols() - column - width;
        let req = factor::qr_in_place_scratch::<usize, R>(height, width, block, Par::Seq, Default::default())
            .or(householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<R>(height, block, trailing));
        if !self
            .scratch
            .as_mut()
            .is_some_and(|s| MemStack::new(s).can_hold(req))
        {
            self.scratch = Some(MemBuffer::new(req));
        }
        let stack = MemStack::new(self.scratch.as_mut().unwrap());
        let view = self.matrix.as_mut().subrows_mut(row, height);
        let (_, rest) = view.split_at_col_mut(column);
        let (mut a, right) = rest.split_at_col_mut(width);
        factor::qr_in_place(
            a.as_mut(),
            self.coefficients.as_mut(),
            &mut self.permutation,
            &mut self.inverse,
            Par::Seq,
            stack,
            Default::default(),
        );
        householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
            a.as_ref(),
            self.coefficients.as_ref(),
            Conj::No,
            right,
            Par::Seq,
            stack,
        );
    }

    fn reduce(
        &mut self,
        removed: usize,
        options: &MarginalizationOptions<R>,
        prior: &mut Prior<R>,
    ) -> Result<(R, usize), SolverError> {
        let n = self.rows.dimension;
        let m = self.rows.values.len() / (n + 1);
        self.matrix.resize_with(m, n + 1, |_, _| R::zero());
        for col in 0..=n {
            for row in 0..m {
                self.matrix[(row, col)] = self.rows.values[row * (n + 1) + col];
            }
        }
        self.qr(0, 0, removed);
        let tolerance = options
            .relative_rank_tolerance
            .unwrap_or_else(|| R::epsilon_impl() * R::from_f64_impl(m.max(removed).max(1) as f64));
        let scale = if m > 0 && removed > 0 {
            self.matrix[(0, 0)].abs()
        } else {
            R::zero()
        };
        let rank = (0..m.min(removed))
            .take_while(|&i| self.matrix[(i, i)].abs() > scale * tolerance)
            .count();
        let separator = n - removed;
        self.qr(rank, removed, separator);
        let rows = (m - rank).min(separator);
        prior.a.resize_with(rows, separator, |_, _| R::zero());
        prior.a.as_mut().fill(R::zero());
        prior.b.resize(rows, R::zero());
        prior.residual.resize(rows, R::zero());
        prior
            .jacobian
            .resize_with(rows, separator, |_, _| R::zero());
        let a = &mut prior.a;
        let b = &mut prior.b;
        for row in 0..rows {
            b[row] = self.matrix[(rank + row, n)];
            for col in row..separator {
                // Restore retained column order; entries below R contain reflectors.
                a[(row, self.permutation[col])] = self.matrix[(rank + row, removed + col)];
            }
        }
        let constant = (rank + rows..m).fold(R::zero(), |sum, row| {
            let r = self.matrix[(row, n)];
            sum + (R::from_f64_impl(0.5) * r) * r
        });
        if !finite(a)
            || b.iter().any(|v| !v.is_finite())
            || !constant.is_finite()
            || (0..m.min(removed)).any(|i| !self.matrix[(i, i)].is_finite())
        {
            return Err(EvaluationError::InvalidEvaluation.into());
        }
        Ok((constant, rank))
    }
}

#[derive(Default)]
struct Planning {
    remove: HashSet<BlockId>,
    full: Layout,
    affected: HashSet<FactorId>,
    touched: HashSet<BlockId>,
    layout: Layout,
    seen: Vec<bool>,
    marks: Vec<usize>,
    columns: Vec<usize>,
    swaps: Vec<(usize, usize)>,
}

pub(crate) struct Marginalizer<R: Real> {
    numeric: Elimination<R>,
}

impl<R: Real> Default for Marginalizer<R> {
    fn default() -> Self {
        Self {
            numeric: Elimination::default(),
        }
    }
}

impl<R: Real> Marginalizer<R> {
    pub(crate) fn run<S, F>(
        &mut self,
        states: &mut S,
        factors: &mut F,
        priors: &mut Priors<R>,
        remove: &[BlockId],
        options: &MarginalizationOptions<R>,
    ) -> Result<MarginalizationReport, SolverError>
    where
        S: StateSchema<Scalar = R>,
        F: FactorSchema<S, Scalar = R>,
    {
        if options
            .relative_rank_tolerance
            .is_some_and(|t| !t.is_finite() || t < R::zero() || t >= R::one())
        {
            return Err(SolverError::InvalidRankTolerance);
        }
        let input = remove;
        let mut planning = Planning::default();
        let Planning {
            remove,
            full,
            affected,
            touched,
            layout,
            seen,
            marks,
            columns,
            swaps,
        } = &mut planning;
        remove.clear();
        remove.extend(input.iter().copied());
        if remove.is_empty() {
            return Ok(MarginalizationReport::default());
        }
        struct Validate<'a> {
            ids: &'a HashSet<BlockId>,
            found: usize,
        }
        impl<R: Real> StateVisitor<R> for Validate<'_> {
            type Error = KeyError;
            fn pool<T: Variable<Scalar = R>>(
                &mut self,
                pool: &mut StatePool<T>,
            ) -> Result<(), KeyError> {
                for &id in self.ids {
                    if let Some(result) = pool.validate(id) {
                        result?;
                        self.found += 1;
                    }
                }
                Ok(())
            }
        }
        let mut validate = Validate {
            ids: remove,
            found: 0,
        };
        states.visit(&mut validate)?;
        if validate.found != remove.len() {
            return Err(KeyError::ForeignSolver.into());
        }

        // ponytail: scan graph metadata and use one dense affected front. Add an
        // incidence directory and sparse fronts when large-window benchmarks need them.
        full.clear();
        states.visit(full)?;
        factors.visit(full)?;
        for (key, prior) in priors.entries.iter() {
            full.factor(FactorId(key), |visit| {
                for (id, _) in &prior.blocks {
                    visit(*id);
                }
            })?;
        }
        affected.clear();
        touched.clear();
        for factor in &full.factors {
            let dependencies = &full.dependencies[factor.dependencies.clone()];
            if dependencies
                .iter()
                .any(|&i| remove.contains(&full.blocks[i].id))
            {
                affected.insert(factor.id);
                touched.extend(dependencies.iter().map(|&i| full.blocks[i].id));
            }
        }
        layout.clear();
        let mut removed_dof = 0;
        for marginalized in [true, false] {
            for block in &full.blocks {
                if remove.contains(&block.id) == marginalized
                    && (marginalized || touched.contains(&block.id))
                {
                    layout.block_index.insert(block.id, layout.blocks.len());
                    layout.blocks.push(Block {
                        id: block.id,
                        offset: layout.dimension,
                        width: block.width,
                    });
                    layout.dimension += block.width;
                }
            }
            if marginalized {
                removed_dof = layout.dimension;
            }
        }
        for factor in &full.factors {
            if affected.contains(&factor.id) {
                layout.factor(factor.id, |visit| {
                    for &i in &full.dependencies[factor.dependencies.clone()] {
                        visit(full.blocks[i].id);
                    }
                })?;
            }
        }
        self.numeric.rows.prepare(layout.dimension)?;
        seen.resize(layout.factors.len(), false);
        seen.fill(false);
        marks.resize(layout.blocks.len(), 0);
        marks.fill(0);
        columns.clear();
        let mut sink = CheckedSink {
            backend: &mut self.numeric.rows,
            layout,
            seen,
            block_marks: marks,
            columns,
            emission: 0,
            allowed: 0..0,
            next_expected: 0,
            active: None,
            failed: false,
        };
        factors.visit(&mut Collect {
            states,
            sink: &mut sink,
            selected: affected,
            swaps,
        })?;
        let absorbed_priors = priors
            .entries
            .iter()
            .filter(|(key, _)| affected.contains(&FactorId(*key)))
            .count();
        for (key, _) in priors.entries.iter() {
            if let Some(&index) = layout.factor_index.get(&FactorId(key)) {
                sink.seen[index] = true;
            }
        }
        if sink.failed || sink.seen.iter().any(|seen| !seen) {
            return Err(EvaluationError::InvalidEmission.into());
        }
        priors.linearize(states, &mut self.numeric.rows, layout, Some(affected))?;
        let prior = Prior::default();
        let new = priors.entries.insert(prior);
        let mut pending = Pending {
            states,
            priors,
            new: Some(new),
            published: false,
        };
        let prior = pending.priors.entries.get_mut(new).expect("pending prior");
        let (constant, rank) = self.numeric.reduce(removed_dof, options, prior)?;
        let report = MarginalizationReport {
            removed_states: remove.len(),
            removed_dof,
            absorbed_factors: affected.len() - absorbed_priors,
            absorbed_priors,
            separator_dof: layout.dimension - removed_dof,
            eliminated_rank: rank,
            prior_rows: prior.a.nrows(),
        };
        prior.blocks.clear();
        prior.blocks.extend(
            layout
                .blocks
                .iter()
                .filter(|block| !remove.contains(&block.id))
                .map(|block| {
                    (
                        block.id,
                        block.offset - removed_dof..block.offset - removed_dof + block.width,
                    )
                }),
        );
        // References are staged under a guard. Failed geometry (or a panic) drops
        // only the unpublished prior and its references, preserving the graph.
        let constant = checked_cost(Ok(pending.priors.constant + constant))?;
        if report.prior_rows > 0 {
            pending.states.visit(&mut CaptureAnchors {
                key: new,
                layout,
                remove,
                offset: removed_dof,
            })?;
        }
        // All fallible numerical work is complete. Storage removal validates IDs
        // already checked above and does not allocate.
        factors.visit(&mut DeleteFactors { ids: affected }).unwrap();
        pending
            .states
            .visit(&mut DeleteStates {
                remove,
                absorbed: affected,
            })
            .unwrap();
        for index in (0..pending.priors.entries.keys.len()).rev() {
            let key = RawKey {
                pool: pending.priors.entries.id,
                local: pending.priors.entries.keys[index],
            };
            if affected.contains(&FactorId(key)) {
                pending.priors.entries.remove(key).expect("live prior");
            }
        }
        pending.priors.constant = constant;
        if report.prior_rows == 0 {
            pending.priors.entries.remove(new).expect("empty prior");
        }
        pending.published = true;
        Ok(report)
    }
}

struct Collect<'a, 'b, 'c, S, R: Real> {
    states: &'a S,
    sink: &'b mut CheckedSink<'c, Rows<R>>,
    selected: &'a HashSet<FactorId>,
    swaps: &'b mut Vec<(usize, usize)>,
}

impl<S, R: Real> FactorVisitor<S, R> for Collect<'_, '_, '_, S, R> {
    type Error = SolverError;
    fn standalone<T: Factor<S, Scalar = R>>(
        &mut self,
        pool: &mut FactorPool<T>,
    ) -> Result<(), SolverError> {
        for (id, factor) in pool.iter() {
            if let Some(&index) = self.sink.layout.factor_index.get(&id) {
                checked_cost(factor.cost(self.states))?;
                self.sink.allowed = index..index + 1;
                self.sink.next_expected = index;
                self.sink
                    .factor(id, |sink| factor.linearize(self.states, sink))?;
            }
        }
        Ok(())
    }
    fn batch<B: FactorBatch<S, Scalar = R>>(
        &mut self,
        pool: &mut BatchPool<B, B::Factor>,
    ) -> Result<(), SolverError> {
        pool.with_selected(self.selected, self.swaps, |model, factors| {
            let indices = factors
                .clone()
                .map(|(id, _)| self.sink.layout.factor_index[&id]);
            let start = indices.clone().min().unwrap();
            let end = indices.max().unwrap() + 1;
            self.sink.allowed = start..end;
            self.sink.next_expected = start;
            checked_cost(model.cost(self.states, factors.clone()))?;
            model.linearize(self.states, factors, self.sink)?;
            Ok(())
        })
    }
}

struct CaptureAnchors<'a> {
    key: RawKey,
    layout: &'a Layout,
    remove: &'a HashSet<BlockId>,
    offset: usize,
}
impl<R: Real> StateVisitor<R> for CaptureAnchors<'_> {
    type Error = SolverError;
    fn pool<T: Variable<Scalar = R>>(
        &mut self,
        pool: &mut StatePool<T>,
    ) -> Result<(), SolverError> {
        let mut anchors = Vec::new();
        for (id, value) in pool.iter() {
            if let Some(&index) = self.layout.block_index.get(&id)
                && !self.remove.contains(&id)
            {
                let inverse = value.inverse();
                if inverse.compose(value).log().iter().any(|v| !v.is_finite()) {
                    return Err(EvaluationError::InvalidEvaluation.into());
                }
                anchors.push(Anchor {
                    prior: self.key,
                    state: StateKey::from_raw(id.0),
                    column: self.layout.blocks[index].offset - self.offset,
                    inverse,
                });
            }
        }
        pool.anchors.extend(anchors);
        Ok(())
    }
}

struct Pending<'a, S: StateSchema> {
    states: &'a mut S,
    priors: &'a mut Priors<S::Scalar>,
    new: Option<RawKey>,
    published: bool,
}
impl<S: StateSchema> Drop for Pending<'_, S> {
    fn drop(&mut self) {
        if !self.published
            && let Some(key) = self.new
        {
            struct Remove(RawKey);
            impl<R: Real> StateVisitor<R> for Remove {
                type Error = Infallible;
                fn pool<T: Variable<Scalar = R>>(
                    &mut self,
                    pool: &mut StatePool<T>,
                ) -> Result<(), Infallible> {
                    pool.anchors.retain(|a| a.prior != self.0);
                    Ok(())
                }
            }
            self.states.visit(&mut Remove(key)).unwrap();
            self.priors.entries.remove(key).expect("unpublished prior");
        }
    }
}

struct DeleteFactors<'a> {
    ids: &'a HashSet<FactorId>,
}
impl<S, R: Real> FactorVisitor<S, R> for DeleteFactors<'_> {
    type Error = Infallible;
    fn standalone<T: Factor<S, Scalar = R>>(
        &mut self,
        pool: &mut FactorPool<T>,
    ) -> Result<(), Infallible> {
        pool.remove_selected(self.ids);
        Ok(())
    }
    fn batch<B: FactorBatch<S, Scalar = R>>(
        &mut self,
        pool: &mut BatchPool<B, B::Factor>,
    ) -> Result<(), Infallible> {
        pool.remove_selected(self.ids);
        Ok(())
    }
}

struct DeleteStates<'a> {
    remove: &'a HashSet<BlockId>,
    absorbed: &'a HashSet<FactorId>,
}
impl<R: Real> StateVisitor<R> for DeleteStates<'_> {
    type Error = Infallible;
    fn pool<T: Variable<Scalar = R>>(&mut self, pool: &mut StatePool<T>) -> Result<(), Infallible> {
        pool.anchors
            .retain(|a| !self.absorbed.contains(&FactorId(a.prior)));
        pool.remove_selected(self.remove);
        Ok(())
    }
}
