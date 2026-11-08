//! Expected taxi states and externally reviewed visual evidence are distinct.
use crate::{Error, planner, world::*};

pub const STAGES: [&str; 14] = [
    "standing",
    "raise_hand",
    "approach",
    "stop",
    "walk_to_door",
    "reach_handle",
    "grasp_handle",
    "opening",
    "open",
    "entering",
    "seated",
    "closing",
    "closed",
    "depart",
];

pub fn expected(stage: usize) -> Result<World, Error> {
    if stage >= STAGES.len() {
        return Err(Error::Invalid("invalid interaction stage"));
    }
    let mut world = planner::plan(planner::CONTROL_PROMPT)?.world;
    for step in 0..=stage {
        let person = match step {
            0 => PersonState::Standing,
            1..=3 => PersonState::Hailing,
            4 => PersonState::Walking,
            5 => PersonState::AtDoor,
            6..=8 => PersonState::Grasping,
            9 => PersonState::Entering,
            _ => PersonState::Seated,
        };
        let door = match step {
            0..=6 => DoorState::Closed,
            7 => DoorState::Opening,
            8..=10 => DoorState::Open,
            11 => DoorState::Closing,
            _ => DoorState::Closed,
        };
        let vehicle = match step {
            0..=1 => VehicleState::Waiting,
            2 => VehicleState::Approaching,
            3..=12 => VehicleState::Stopped,
            _ => VehicleState::Departing,
        };
        let car_x = match step {
            0..=1 => 10000,
            2 => 5000,
            3..=12 => 1000,
            _ => 20000,
        };
        let person_x = if step < 4 {
            0
        } else if step < 13 {
            1000
        } else {
            car_x
        };
        let mut relations = vec![Relation {
            from: ObjectId(3),
            to: ObjectId(2),
            kind: RelationKind::PartOf,
        }];
        if (4..=9).contains(&step) {
            relations.push(Relation {
                from: ObjectId(1),
                to: ObjectId(3),
                kind: RelationKind::NearDoor,
            });
        }
        if (5..=8).contains(&step) {
            relations.push(Relation {
                from: ObjectId(1),
                to: ObjectId(3),
                kind: RelationKind::HandNearHandle,
            });
        }
        if (6..=8).contains(&step) {
            relations.push(Relation {
                from: ObjectId(1),
                to: ObjectId(3),
                kind: RelationKind::Contact,
            });
        }
        if step == 9 {
            relations.push(Relation {
                from: ObjectId(1),
                to: ObjectId(3),
                kind: RelationKind::OccludedBy,
            });
        }
        if step >= 10 {
            relations.push(Relation {
                from: ObjectId(1),
                to: ObjectId(2),
                kind: RelationKind::Inside,
            });
        }
        let updates = [
            Update {
                id: ObjectId(1),
                position: [person_x, 0, 0],
                pose: if step >= 10 {
                    Pose::Seated
                } else if step < 4 && step > 0 {
                    Pose::Hailing
                } else {
                    Pose::Rest
                },
            },
            Update {
                id: ObjectId(2),
                position: [car_x, 0, 0],
                pose: Pose::Rest,
            },
            Update {
                id: ObjectId(3),
                position: [car_x, 0, 0],
                pose: if door == DoorState::Open {
                    Pose::Open
                } else {
                    Pose::Rest
                },
            },
        ];
        world.apply_states(
            2 + step as u64,
            &updates,
            &relations,
            &[
                (ObjectId(1), ObjectState::Person(person)),
                (ObjectId(2), ObjectState::Vehicle(vehicle)),
                (ObjectId(3), ObjectState::Door(door)),
            ],
        )?;
    }
    Ok(world)
}

#[derive(Clone, Debug)]
pub struct ReferenceFrame {
    pub snapshot: Snapshot,
    pub digest: String,
}

/// A review is supplied by an external visual evaluator, never inferred from prompts.
pub struct VisualReview {
    pub digest: String,
    pub action: bool,
    pub person: bool,
    pub vehicle: bool,
    pub background: bool,
    pub transition: bool,
    pub evidence: String,
}

#[derive(Default)]
pub struct ReferenceFrameManager {
    canonical: Option<ReferenceFrame>,
    latest: Option<ReferenceFrame>,
}

impl ReferenceFrameManager {
    pub fn canonical(&self) -> Option<&ReferenceFrame> {
        self.canonical.as_ref()
    }
    pub fn latest(&self) -> Option<&ReferenceFrame> {
        self.latest.as_ref()
    }
    pub fn accept(&mut self, frame: ReferenceFrame, review: &VisualReview) -> Result<(), Error> {
        if frame.digest.len() != 64
            || !frame.digest.bytes().all(|b| b.is_ascii_hexdigit())
            || frame.digest != review.digest
            || review.evidence.trim().is_empty()
            || review.evidence.len() > 4096
            || ![
                review.action,
                review.person,
                review.vehicle,
                review.background,
                review.transition,
            ]
            .into_iter()
            .all(|v| v)
        {
            return Err(Error::Invalid(
                "visual evidence missing, failed or mismatched",
            ));
        }
        if let Some(previous) = &self.latest {
            let ids = |s: &Snapshot| {
                s.objects()
                    .iter()
                    .map(|o| (o.id, o.kind, o.appearance))
                    .collect::<Vec<_>>()
            };
            if frame.snapshot.tick() != previous.snapshot.tick() + 1
                || ids(&frame.snapshot) != ids(&previous.snapshot)
                || frame.snapshot.globals() != previous.snapshot.globals()
            {
                return Err(Error::Invalid(
                    "reference provenance changed or stage skipped",
                ));
            }
        } else if frame.snapshot.tick() != 2 {
            return Err(Error::Invalid(
                "canonical reference must be the initial stage",
            ));
        }
        if self.canonical.is_none() {
            self.canonical = Some(frame.clone());
        }
        self.latest = Some(frame);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_door_chain_and_spatial_relations() {
        let chain =
            [6, 7, 8, 11, 12].map(|s| expected(s).unwrap().current().state(ObjectId(3)).unwrap());
        assert_eq!(
            chain,
            [
                DoorState::Closed,
                DoorState::Opening,
                DoorState::Open,
                DoorState::Closing,
                DoorState::Closed
            ]
            .map(ObjectState::Door)
        );
        for (stage, kind) in [
            (4, RelationKind::NearDoor),
            (5, RelationKind::HandNearHandle),
            (9, RelationKind::OccludedBy),
            (10, RelationKind::Inside),
        ] {
            assert!(
                expected(stage)
                    .unwrap()
                    .current()
                    .relations()
                    .iter()
                    .any(|r| r.kind == kind)
            );
        }
        let a = expected(0).unwrap();
        let b = expected(13).unwrap();
        for (a, b) in a.current().objects().iter().zip(b.current().objects()) {
            assert_eq!((a.id, a.appearance, a.kind), (b.id, b.appearance, b.kind));
        }
        assert!(expected(14).is_err());
    }
    #[test]
    fn invalid_door_jump_is_atomic_and_state_changes_are_detected() {
        let mut w = expected(6).unwrap();
        let before = w.clone();
        assert!(
            w.apply_states(
                99,
                &[],
                w.current().relations().to_vec().as_slice(),
                &[(ObjectId(3), ObjectState::Door(DoorState::Open))]
            )
            .is_err()
        );
        assert_eq!(w, before);
        w.apply_states(
            99,
            &[],
            &[],
            &[(ObjectId(3), ObjectState::Door(DoorState::Opening))],
        )
        .unwrap();
        assert!(
            crate::adaptive::detect(before.current(), w.current(), &[], false)
                .affected
                .contains(&ObjectId(3))
        );
    }
    #[test]
    fn failed_or_stale_review_cannot_advance_or_replace_identity_anchor() {
        let mut manager = ReferenceFrameManager::default();
        let frame = |s| ReferenceFrame {
            snapshot: expected(s).unwrap().current().clone(),
            digest: format!("{s:064x}"),
        };
        let mut review = VisualReview {
            digest: frame(0).digest,
            action: true,
            person: true,
            vehicle: true,
            background: true,
            transition: true,
            evidence: "inspected source frames".into(),
        };
        manager.accept(frame(0), &review).unwrap();
        assert!(manager.accept(frame(1), &review).is_err());
        review.digest = frame(1).digest;
        review.action = false;
        assert!(manager.accept(frame(1), &review).is_err());
        assert_eq!(manager.latest().unwrap().snapshot.tick(), 2);
        review.action = true;
        manager.accept(frame(1), &review).unwrap();
        assert_eq!(manager.canonical().unwrap().snapshot.tick(), 2);
        review.digest = frame(3).digest;
        assert!(manager.accept(frame(3), &review).is_err());
    }
}
