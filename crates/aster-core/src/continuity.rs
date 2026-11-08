//! Decide whether previous neural reference pixels may condition the next event.
//! Temporal reference reuse does not imply spatially sparse transformer execution.
use crate::{
    adaptive::Changes,
    world::{Globals, ObjectId, Snapshot},
};

#[derive(Clone, Debug)]
pub struct ReferenceStamp {
    pub tick: u64,
    pub identities: Vec<(ObjectId, u64)>,
    pub globals: Globals,
}

impl ReferenceStamp {
    pub fn for_snapshot(snapshot: &Snapshot) -> Self {
        Self {
            tick: snapshot.tick(),
            identities: snapshot
                .objects()
                .iter()
                .map(|o| (o.id, o.appearance))
                .collect(),
            globals: snapshot.globals(),
        }
    }
    pub fn can_condition(&self, before: &Snapshot, changes: &Changes) -> bool {
        !changes.global
            && self.tick == before.tick()
            && self.globals == before.globals()
            && self.identities
                == before
                    .objects()
                    .iter()
                    .map(|o| (o.id, o.appearance))
                    .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adaptive, planner};
    #[test]
    fn stale_identity_and_global_changes_invalidate_reference() {
        let mut p = planner::plan(planner::CONTROL_PROMPT).unwrap();
        let before = p.world.current().clone();
        let mut reference = ReferenceStamp::for_snapshot(&before);
        let e = &p.events[0];
        p.world
            .apply_scene(e.tick, &e.updates, Some(&e.relations), None)
            .unwrap();
        let mut changes = adaptive::detect(&before, p.world.current(), &[], false);
        assert!(reference.can_condition(&before, &changes));
        assert!(!reference.can_condition(p.world.current(), &changes));
        changes.global = true;
        assert!(!reference.can_condition(&before, &changes));
        changes.global = false;
        reference.identities[0].1 += 1;
        assert!(!reference.can_condition(&before, &changes));
    }
}
