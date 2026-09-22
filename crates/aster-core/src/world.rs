//! Deterministic symbolic state; appearance handles are not generated images.
use crate::Error;
use std::collections::VecDeque;

pub const MAX_OBJECTS: usize = 256;
pub const MAX_HISTORY: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Person,
    Vehicle,
    Environment,
    Door,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pose {
    Rest,
    Hailing,
    Seated,
    Open,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationKind {
    PartOf,
    Contact,
    Inside,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Relation {
    pub from: ObjectId,
    pub to: ObjectId,
    pub kind: RelationKind,
}

/// Opaque lighting/environment asset revisions; camera position in millimetres.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Globals {
    pub camera: [i32; 3],
    pub lighting: u64,
    pub environment: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Object {
    pub id: ObjectId,
    pub kind: Kind,
    /// Stable asset reference, not a guarantee of neural identity preservation.
    pub appearance: u64,
    /// Fixed-point world coordinates in millimetres.
    pub position: [i32; 3],
    pub pose: Pose,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Update {
    pub id: ObjectId,
    pub position: [i32; 3],
    pub pose: Pose,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    tick: u64,
    objects: Vec<Object>,
    relations: Vec<Relation>,
    globals: Globals,
}

impl Snapshot {
    pub fn relations(&self) -> &[Relation] {
        &self.relations
    }

    pub fn globals(&self) -> Globals {
        self.globals
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct World {
    current: Snapshot,
    history: VecDeque<Snapshot>,
    history_limit: usize,
}

impl World {
    /// IDs are unique and immutable for this world's lifetime. No implicit spawns.
    pub fn new(mut objects: Vec<Object>, history_limit: usize) -> Result<Self, Error> {
        if objects.is_empty() || objects.len() > MAX_OBJECTS || history_limit > MAX_HISTORY {
            return Err(Error::Invalid("world object/history limit exceeded"));
        }
        objects.sort_by_key(|o| o.id.0);
        if objects.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return Err(Error::Invalid("duplicate object ID"));
        }
        Ok(Self {
            current: Snapshot {
                tick: 0,
                objects,
                relations: Vec::new(),
                globals: Globals::default(),
            },
            history: VecDeque::new(),
            history_limit,
        })
    }

    pub fn current(&self) -> &Snapshot {
        &self.current
    }

    pub fn history(&self) -> &VecDeque<Snapshot> {
        &self.history
    }

    /// All participants change together. Invalid events leave state/history intact.
    /// Empty events advance time explicitly; ticks must increase strictly.
    pub fn apply(&mut self, tick: u64, updates: &[Update]) -> Result<(), Error> {
        self.apply_scene(tick, updates, None, None)
    }

    /// Relationships are replaced as a complete bounded set, in the same transaction.
    pub fn apply_scene(
        &mut self,
        tick: u64,
        updates: &[Update],
        relations: Option<&[Relation]>,
        globals: Option<Globals>,
    ) -> Result<(), Error> {
        if tick <= self.current.tick || updates.len() > MAX_OBJECTS {
            return Err(Error::Invalid("invalid event tick or participant count"));
        }
        for (index, update) in updates.iter().enumerate() {
            if updates[..index].iter().any(|u| u.id == update.id)
                || !self.current.objects.iter().any(|o| o.id == update.id)
            {
                return Err(Error::Invalid("duplicate or unknown event participant"));
            }
        }
        if let Some(relations) = relations {
            if relations.len() > MAX_OBJECTS {
                return Err(Error::Invalid("relationship limit exceeded"));
            }
            for (index, relation) in relations.iter().enumerate() {
                if relation.from == relation.to
                    || relations[..index].contains(relation)
                    || [relation.from, relation.to]
                        .iter()
                        .any(|id| !self.current.objects.iter().any(|o| o.id == *id))
                {
                    return Err(Error::Invalid("invalid relationship"));
                }
            }
        }
        let mut next = self.current.clone();
        if let Some(relations) = relations {
            next.relations = relations.to_vec();
        }
        if let Some(globals) = globals {
            next.globals = globals;
        }
        next.tick = tick;
        for update in updates {
            // Membership checked above; mutate only dynamic fields.
            for object in &mut next.objects {
                if object.id == update.id {
                    object.position = update.position;
                    object.pose = update.pose;
                }
            }
        }
        let previous = std::mem::replace(&mut self.current, next);
        if self.history_limit != 0 {
            if self.history.len() == self.history_limit {
                self.history.pop_front();
            }
            self.history.push_back(previous);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(id: u32) -> Object {
        Object {
            id: ObjectId(id),
            kind: Kind::Person,
            appearance: u64::from(id) + 100,
            position: [0; 3],
            pose: Pose::Rest,
        }
    }

    fn update(id: u32) -> Update {
        Update {
            id: ObjectId(id),
            position: [10, 0, 0],
            pose: Pose::Hailing,
        }
    }

    #[test]
    fn identity_history_and_joint_event() {
        let mut world = World::new(vec![object(2), object(1)], 2).unwrap();
        world.apply(1, &[update(1), update(2)]).unwrap();
        assert_eq!(world.current.objects[0].appearance, 101);
        assert_eq!(world.current.objects[1].appearance, 102);
        assert!(
            world
                .current
                .objects
                .iter()
                .all(|o| o.position == [10, 0, 0])
        );
        assert!(
            world.history[0]
                .objects
                .iter()
                .all(|o| o.position == [0; 3])
        );
        for tick in 2..100 {
            world.apply(tick, &[]).unwrap();
        }
        assert_eq!(world.history.len(), 2);
        assert_eq!(world.history[0].tick, 97);
        assert_eq!(world.current.tick, 99);
    }

    #[test]
    fn invalid_events_are_atomic() {
        let mut world = World::new(vec![object(1)], 2).unwrap();
        for (tick, updates) in [
            (0, vec![]),
            (1, vec![update(1), update(99)]),
            (1, vec![update(1), update(1)]),
        ] {
            let before = world.current.clone();
            assert!(world.apply(tick, &updates).is_err());
            assert_eq!(world.current, before);
            assert!(world.history.is_empty());
        }
        world.apply(2, &[update(1)]).unwrap();
        assert!(world.apply(1, &[]).is_err());
    }

    #[test]
    fn limits_and_unchanged_objects() {
        assert!(World::new(vec![], 1).is_err());
        assert!(World::new(vec![object(1), object(1)], 1).is_err());
        assert!(World::new(vec![object(1)], MAX_HISTORY + 1).is_err());
        assert!(World::new((0..=MAX_OBJECTS as u32).map(object).collect(), 1).is_err());
        let mut world = World::new(vec![object(1), object(2)], 0).unwrap();
        world.apply(u64::MAX, &[update(1)]).unwrap();
        assert_eq!(world.current.objects[1], object(2));
        assert!(world.history.is_empty());
        assert!(world.apply(u64::MAX, &[]).is_err());
    }

    #[test]
    fn relationships_and_globals_commit_together() {
        let mut world = World::new(vec![object(1), object(2)], 2).unwrap();
        let relation = Relation {
            from: ObjectId(1),
            to: ObjectId(2),
            kind: RelationKind::Contact,
        };
        let globals = Globals {
            lighting: 5,
            ..Globals::default()
        };
        world
            .apply_scene(1, &[update(1)], Some(&[relation]), Some(globals))
            .unwrap();
        let before = world.current.clone();
        let invalid = Relation {
            to: ObjectId(99),
            ..relation
        };
        assert!(
            world
                .apply_scene(2, &[update(2)], Some(&[invalid]), Some(Globals::default()))
                .is_err()
        );
        assert_eq!(world.current, before);
        assert_eq!(world.history.len(), 1);
        assert_eq!(world.current.globals(), globals);
        assert_eq!(world.current.relations(), &[relation]);
    }
}
