//! Conservative planning only. This module does not skip neural operations itself.
use crate::world::{ObjectId, Snapshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub min: [i32; 2],
    pub max: [i32; 2],
}

impl Rect {
    fn valid(self) -> bool {
        self.min[0] < self.max[0] && self.min[1] < self.max[1]
    }
    fn union(self, other: Self) -> Self {
        Self {
            min: [self.min[0].min(other.min[0]), self.min[1].min(other.min[1])],
            max: [self.max[0].max(other.max[0]), self.max[1].max(other.max[1])],
        }
    }
}

/// Renderer-supplied bounds in pixels, covering silhouette AND all indirect effects.
pub struct Influence {
    pub id: ObjectId,
    pub before: Rect,
    pub after: Rect,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Changes {
    pub affected: Vec<ObjectId>,
    pub region: Option<Rect>,
    pub global: bool,
    pub unknown_influence: bool,
}

pub fn detect(
    before: &Snapshot,
    after: &Snapshot,
    influences: &[Influence],
    complete: bool,
) -> Changes {
    let mut ids = Vec::new();
    for object in before.objects().iter().chain(after.objects()) {
        let old = before.objects().iter().find(|o| o.id == object.id);
        let new = after.objects().iter().find(|o| o.id == object.id);
        if old != new && !ids.contains(&object.id) {
            ids.push(object.id);
        }
    }
    for relation in before.relations().iter().chain(after.relations()) {
        if before.relations().contains(relation) != after.relations().contains(relation) {
            for id in [relation.from, relation.to] {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
    }
    // Undirected conservative closure includes removed edges and newly introduced edges.
    loop {
        let count = ids.len();
        for relation in before.relations().iter().chain(after.relations()) {
            if ids.contains(&relation.from) || ids.contains(&relation.to) {
                for id in [relation.from, relation.to] {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }
        if ids.len() == count {
            break;
        }
    }
    ids.sort_by_key(|id| id.0);
    let mut unknown = !complete;
    let mut region: Option<Rect> = None;
    for id in &ids {
        let mut matching = influences.iter().filter(|i| i.id == *id);
        match (matching.next(), matching.next()) {
            (Some(i), None) if i.before.valid() && i.after.valid() => {
                let bounds = i.before.union(i.after);
                region = Some(region.map_or(bounds, |r| r.union(bounds)));
            }
            _ => unknown = true,
        }
    }
    Changes {
        affected: ids,
        region,
        global: before.globals() != after.globals(),
        unknown_influence: unknown,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Reuse,
    Local,
    Regional,
    Full,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub local: bool,
    pub regional: bool,
}

/// Exact caller-owned conditioning identity: model/settings/prompt/time/seed/version.
/// A cache implementation must invalidate entries when any of these changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheKey {
    pub model_revision: String,
    pub conditioning: String,
    pub seed: u64,
    pub frame: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Decision {
    pub mode: Mode,
    pub reason: &'static str,
}

pub fn select(
    changes: &Changes,
    capabilities: Capabilities,
    cached: Option<&CacheKey>,
    requested: &CacheKey,
) -> Decision {
    let (mode, reason) = if changes.global || changes.unknown_influence {
        (Mode::Full, "global_or_unknown_influence")
    } else if cached != Some(requested) {
        (Mode::Full, "missing_or_changed_cache_conditioning")
    } else if changes.affected.is_empty() {
        (Mode::Reuse, "unchanged_state_and_conditioning")
    } else if changes.region.is_none() {
        (Mode::Full, "missing_region")
    } else if changes.affected.len() == 1 && capabilities.local {
        (Mode::Local, "isolated_change_supported")
    } else if capabilities.regional {
        (Mode::Regional, "coupled_change_supported")
    } else {
        (Mode::Full, "backend_does_not_support_partial_generation")
    };
    Decision { mode, reason }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        planner,
        world::{Globals, Pose, Update},
    };

    #[test]
    fn closure_and_old_new_bounds_cover_revealed_background() {
        let mut p = planner::plan(planner::TAXI_PROMPT).unwrap();
        let before = p.world.current().clone();
        p.world
            .apply(
                2,
                &[Update {
                    id: ObjectId(2),
                    position: [5, 0, 0],
                    pose: Pose::Rest,
                }],
            )
            .unwrap();
        let bounds = [2, 3].map(|id| Influence {
            id: ObjectId(id),
            before: Rect {
                min: [0, 0],
                max: [10, 10],
            },
            after: Rect {
                min: [20, 0],
                max: [30, 10],
            },
        });
        let c = detect(&before, p.world.current(), &bounds, true);
        assert_eq!(c.affected, [ObjectId(2), ObjectId(3)]);
        assert_eq!(
            c.region.unwrap(),
            Rect {
                min: [0, 0],
                max: [30, 10]
            }
        );
        assert!(!c.unknown_influence);
        assert!(detect(&before, p.world.current(), &bounds[..1], true).unknown_influence);
        p.world
            .apply_scene(3, &[], None, Some(Globals::default()))
            .unwrap();
        assert!(detect(&before, p.world.current(), &bounds, true).global);
    }

    #[test]
    fn all_modes_require_valid_cache_and_backend_capability() {
        let key = CacheKey {
            model_revision: "r1".into(),
            conditioning: "p/settings".into(),
            seed: 42,
            frame: 1,
        };
        let caps = Capabilities {
            local: true,
            regional: true,
        };
        let mut c = Changes {
            affected: vec![],
            region: None,
            global: false,
            unknown_influence: false,
        };
        assert_eq!(select(&c, caps, Some(&key), &key).mode, Mode::Reuse);
        c.affected.push(ObjectId(1));
        c.region = Some(Rect {
            min: [0, 0],
            max: [10, 10],
        });
        assert_eq!(select(&c, caps, Some(&key), &key).mode, Mode::Local);
        c.affected.push(ObjectId(2));
        assert_eq!(select(&c, caps, Some(&key), &key).mode, Mode::Regional);
        assert_eq!(
            select(&c, Capabilities::default(), Some(&key), &key).mode,
            Mode::Full
        );
        assert_eq!(select(&c, caps, None, &key).mode, Mode::Full);
        let changed = CacheKey {
            frame: 2,
            ..key.clone()
        };
        assert_eq!(select(&c, caps, Some(&key), &changed).mode, Mode::Full);
        c.unknown_influence = true;
        assert_eq!(select(&c, caps, Some(&key), &key).mode, Mode::Full);
    }
}
