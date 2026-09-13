use crate::{
    KeyError,
    key::{LocalKey, PoolId, RawKey},
};

const VACANT: u32 = u32::MAX;

struct Slot {
    generation: u32,
    dense_index: u32,
}

/// Packed payloads and local identities, with checked generational indirection.
/// No per-entry allocations; removal moves at most one surviving payload.
pub(crate) struct DensePool<T> {
    pub(crate) id: PoolId,
    pub(crate) values: Vec<T>,
    pub(crate) keys: Vec<LocalKey>,
    slots: Vec<Slot>,
    free_slots: Vec<u32>,
}

impl<T> Default for DensePool<T> {
    fn default() -> Self {
        Self {
            id: PoolId::new(),
            values: Vec::new(),
            keys: Vec::new(),
            slots: Vec::new(),
            free_slots: Vec::new(),
        }
    }
}

impl<T> DensePool<T> {
    pub(crate) fn reserve(&mut self, additional: usize) {
        let live = self
            .values
            .len()
            .checked_add(additional)
            .expect("pool capacity overflow");
        let slots = self
            .slots
            .len()
            .checked_add(additional.saturating_sub(self.free_slots.len()))
            .expect("pool slot capacity overflow");
        assert!(
            live <= VACANT as usize && slots <= VACANT as usize,
            "pool index space exhausted"
        );
        // Reserve every parallel buffer before publishing any new entry. The free
        // list can hold every slot, so deleting entries never grows it.
        self.values.reserve(additional);
        self.keys.reserve(additional);
        self.slots.reserve(slots - self.slots.len());
        self.free_slots.reserve(slots - self.free_slots.len());
    }

    pub(crate) fn insert(&mut self, value: T) -> RawKey {
        self.reserve(1);
        let dense_index = self.values.len() as u32;
        let slot = match self.free_slots.pop() {
            Some(slot) => {
                self.slots[slot as usize].dense_index = dense_index;
                slot
            }
            None => {
                let slot = self.slots.len() as u32;
                self.slots.push(Slot {
                    generation: 0,
                    dense_index,
                });
                slot
            }
        };
        let local = LocalKey {
            slot,
            generation: self.slots[slot as usize].generation,
        };
        self.values.push(value);
        self.keys.push(local);
        RawKey {
            pool: self.id,
            local,
        }
    }

    pub(crate) fn index(&self, key: RawKey) -> Result<usize, KeyError> {
        if key.pool != self.id {
            return Err(KeyError::ForeignSolver);
        }
        let slot = self
            .slots
            .get(key.local.slot as usize)
            .ok_or(KeyError::Unknown)?;
        if slot.generation != key.local.generation || slot.dense_index == VACANT {
            return Err(KeyError::Stale);
        }
        Ok(slot.dense_index as usize)
    }

    pub(crate) fn get(&self, key: RawKey) -> Result<&T, KeyError> {
        Ok(&self.values[self.index(key)?])
    }

    pub(crate) fn get_mut(&mut self, key: RawKey) -> Result<&mut T, KeyError> {
        let index = self.index(key)?;
        Ok(&mut self.values[index])
    }

    // Only live, internally maintained batch-directory links use these methods.
    // Batch deletion must invalidate its payload directory before reusing a slot.
    pub(crate) fn at_slot(&self, slot: u32) -> &T {
        &self.values[self.slots[slot as usize].dense_index as usize]
    }

    pub(crate) fn at_slot_mut(&mut self, slot: u32) -> &mut T {
        let index = self.slots[slot as usize].dense_index as usize;
        &mut self.values[index]
    }

    pub(crate) fn remove(&mut self, key: RawKey) -> Result<T, KeyError> {
        let index = self.index(key)?;
        let value = self.values.swap_remove(index);
        self.keys.swap_remove(index);
        if let Some(moved) = self.keys.get(index) {
            self.slots[moved.slot as usize].dense_index = index as u32;
        }
        let slot = &mut self.slots[key.local.slot as usize];
        slot.dense_index = VACANT;
        if let Some(generation) = slot.generation.checked_add(1) {
            slot.generation = generation;
            self.free_slots.push(key.local.slot);
        }
        // Exhausted generations stay vacant forever. Drop the value only after
        // metadata repair, when ownership passes back to the caller.
        Ok(value)
    }

    pub(crate) fn iter(&self) -> impl ExactSizeIterator<Item = (RawKey, &T)> {
        self.keys
            .iter()
            .copied()
            .zip(&self.values)
            .map(|(local, value)| {
                (
                    RawKey {
                        pool: self.id,
                        local,
                    },
                    value,
                )
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    fn check<T>(pool: &DensePool<T>) {
        assert_eq!(pool.values.len(), pool.keys.len());
        let mut seen = vec![false; pool.slots.len()];
        for (index, &local) in pool.keys.iter().enumerate() {
            assert!(!seen[local.slot as usize]);
            seen[local.slot as usize] = true;
            let raw = RawKey {
                pool: pool.id,
                local,
            };
            assert_eq!(pool.index(raw).unwrap(), index);
        }
        for &slot in &pool.free_slots {
            assert!(!seen[slot as usize]);
            seen[slot as usize] = true;
            assert_eq!(pool.slots[slot as usize].dense_index, VACANT);
        }
        for (slot, seen) in pool.slots.iter().zip(seen) {
            if !seen {
                assert_eq!(slot.generation, u32::MAX);
                assert_eq!(slot.dense_index, VACANT);
            }
        }
    }

    #[test]
    fn compaction_reuse_and_reserved_churn_preserve_invariants() {
        let mut pool = DensePool::default();
        pool.reserve(128);
        let capacity = (
            pool.values.capacity(),
            pool.keys.capacity(),
            pool.slots.capacity(),
            pool.free_slots.capacity(),
        );
        let mut live = Vec::new();
        let mut seed = 7_u64;
        for value in 0..10_000 {
            if live.is_empty() || (live.len() < 100 && value % 3 != 0) {
                live.push((pool.insert(value), value));
            } else {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let index = (seed as usize) % live.len();
                let (key, expected) = live.swap_remove(index);
                assert_eq!(pool.remove(key).unwrap(), expected);
                assert!(matches!(pool.get(key), Err(KeyError::Stale)));
                assert!(matches!(pool.remove(key), Err(KeyError::Stale)));
            }
            for &(key, value) in &live {
                assert_eq!(*pool.get(key).unwrap(), value);
            }
            check(&pool);
        }
        assert_eq!(
            capacity,
            (
                pool.values.capacity(),
                pool.keys.capacity(),
                pool.slots.capacity(),
                pool.free_slots.capacity(),
            )
        );
    }

    #[test]
    fn invalid_keys_and_exhaustion_cannot_alias_live_values() {
        let mut pool = DensePool::default();
        let first = pool.insert(10);
        let other = DensePool::<i32>::default();
        let foreign = RawKey {
            pool: other.id,
            ..first
        };
        let unknown = RawKey {
            local: LocalKey {
                slot: u32::MAX,
                generation: 0,
            },
            ..first
        };
        assert!(matches!(pool.get(foreign), Err(KeyError::ForeignSolver)));
        assert!(matches!(pool.get(unknown), Err(KeyError::Unknown)));
        assert!(matches!(pool.remove(foreign), Err(KeyError::ForeignSolver)));
        assert!(matches!(pool.remove(unknown), Err(KeyError::Unknown)));

        // Simulate the final usable generation without billions of insertions.
        pool.slots[first.local.slot as usize].generation = u32::MAX;
        pool.keys[0].generation = u32::MAX;
        let last = RawKey {
            local: pool.keys[0],
            ..first
        };
        assert!(matches!(pool.get(first), Err(KeyError::Stale)));
        assert_eq!(pool.remove(last).unwrap(), 10);
        let replacement = pool.insert(20);
        assert_ne!(replacement.local.slot, last.local.slot);
        assert!(matches!(pool.get(last), Err(KeyError::Stale)));
        assert_eq!(*pool.get(replacement).unwrap(), 20);
        check(&pool);

        let capacity = pool.values.capacity();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.reserve(usize::MAX);
            }))
            .is_err()
        );
        assert_eq!(pool.values.capacity(), capacity);
        assert_eq!(*pool.get(replacement).unwrap(), 20);
        check(&pool);
    }

    #[test]
    fn values_are_dropped_exactly_once_after_removal_or_pool_drop() {
        struct Value(Rc<Cell<usize>>);
        impl Drop for Value {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let drops = Rc::new(Cell::new(0));
        let mut pool = DensePool::default();
        let first = pool.insert(Value(drops.clone()));
        pool.insert(Value(drops.clone()));
        let removed = pool.remove(first).unwrap();
        check(&pool);
        assert_eq!(drops.get(), 0);
        drop(removed);
        assert_eq!(drops.get(), 1);
        pool.insert(Value(drops.clone()));
        drop(pool);
        assert_eq!(drops.get(), 3);
    }
}
