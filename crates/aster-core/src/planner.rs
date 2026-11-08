//! Explicit restricted grammar. Unrecognized prose is never silently replaced by a taxi clip.
use crate::Error;
use crate::world::{Globals, Kind, Object, ObjectId, Pose, Relation, RelationKind, Update, World};

pub const MAX_PROMPT_BYTES: usize = 16 * 1024;
pub const TAXI_PROMPT: &str = "Человек стоит на улице, останавливает такси, открывает дверь и садится в машину. Такси уезжает";
pub const CONTROL_PROMPT: &str = "Человек стоит на улице, поднимает руку и останавливает жёлтое такси. Такси подъезжает и останавливается. Человек открывает заднюю дверь, садится в машину. Такси уезжает";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub tick: u64,
    pub action: &'static str,
    pub updates: Vec<Update>,
    pub relations: Vec<Relation>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub prompt: String,
    pub world: World,
    pub events: Vec<Event>,
}

pub fn validate_prompt(prompt: &str) -> Result<(), Error> {
    if prompt.trim().is_empty()
        || prompt.len() > MAX_PROMPT_BYTES
        || prompt
            .chars()
            .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
    {
        return Err(Error::Invalid(
            "prompt must contain 1..16384 UTF-8 bytes without control characters",
        ));
    }
    Ok(())
}

/// Grammar accepts the documented taxi sentences, with optional approach/door-close.
/// Coordinates and timing are explicit planner defaults, not inferred physical truth.
pub fn plan(prompt: &str) -> Result<Plan, Error> {
    validate_prompt(prompt)?;
    let normalized = prompt
        .to_lowercase()
        .replace(" и ", ",")
        .replace(
            "поднимает руку,останавливает жёлтое такси",
            "останавливает такси",
        )
        .replace(
            "поднимает руку,останавливает желтое такси",
            "останавливает такси",
        )
        .replace("такси подъезжает,останавливается", "такси подъезжает")
        .replace("открывает заднюю дверь", "открывает дверь");
    let clauses: Vec<_> = normalized
        .split([',', '.', ';', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if clauses.first() != Some(&"человек стоит на улице") || clauses.len() > 8 {
        return Err(Error::Invalid(
            "unsupported scene grammar; use the documented taxi scenario",
        ));
    }
    let objects = [
        (1, Kind::Person, [0, 0, 0]),
        (2, Kind::Vehicle, [10000, 0, 0]),
        (3, Kind::Door, [10000, 0, 0]),
        (4, Kind::Environment, [0, 0, 0]),
    ]
    .map(|(id, kind, position)| Object {
        id: ObjectId(id),
        kind,
        appearance: u64::from(id),
        position,
        pose: Pose::Rest,
    });
    let mut world = World::new(objects.to_vec(), 8)?;
    let part = Relation {
        from: ObjectId(3),
        to: ObjectId(2),
        kind: RelationKind::PartOf,
    };
    world.apply_scene(
        1,
        &[],
        Some(&[part]),
        Some(Globals {
            environment: 1,
            lighting: 1,
            camera: [0, 2000, -5000],
        }),
    )?;
    let mut events = Vec::new();
    let mut phase = 0;
    let mut door_closed = false;
    let update = |id, x, pose| Update {
        id: ObjectId(id),
        position: [x, 0, 0],
        pose,
    };
    let mut push = |action, updates, relations| {
        events.push(Event {
            tick: 1 + (events.len() as u64 + 1) * 24,
            action,
            updates,
            relations,
        });
    };
    for clause in &clauses[1..] {
        match (*clause, phase) {
            ("останавливает такси" | "человек останавливает такси", 0) =>
            {
                push("hail", vec![update(1, 0, Pose::Hailing)], vec![part]);
                phase = 1;
            }
            ("такси подъезжает", 1) => {
                push(
                    "approach",
                    vec![update(2, 1000, Pose::Rest), update(3, 1000, Pose::Rest)],
                    vec![part],
                );
                phase = 2;
            }
            ("открывает дверь" | "человек открывает дверь", 1 | 2) =>
            {
                if phase == 1 {
                    push(
                        "approach",
                        vec![update(2, 1000, Pose::Rest), update(3, 1000, Pose::Rest)],
                        vec![part],
                    );
                }
                push(
                    "open_door",
                    vec![update(1, 1000, Pose::Rest), update(3, 1000, Pose::Open)],
                    vec![
                        part,
                        Relation {
                            from: ObjectId(1),
                            to: ObjectId(3),
                            kind: RelationKind::Contact,
                        },
                    ],
                );
                phase = 3;
            }
            ("садится в машину" | "человек садится в машину", 3) =>
            {
                push(
                    "enter",
                    vec![update(1, 1000, Pose::Seated)],
                    vec![
                        part,
                        Relation {
                            from: ObjectId(1),
                            to: ObjectId(2),
                            kind: RelationKind::Inside,
                        },
                    ],
                );
                phase = 4;
            }
            ("закрывает дверь" | "человек закрывает дверь", 4) if !door_closed =>
            {
                push(
                    "close_door",
                    vec![update(3, 1000, Pose::Rest)],
                    vec![
                        part,
                        Relation {
                            from: ObjectId(1),
                            to: ObjectId(2),
                            kind: RelationKind::Inside,
                        },
                    ],
                );
                door_closed = true;
            }
            ("такси уезжает", 4) => {
                if !door_closed {
                    push(
                        "close_door",
                        vec![update(3, 1000, Pose::Rest)],
                        vec![
                            part,
                            Relation {
                                from: ObjectId(1),
                                to: ObjectId(2),
                                kind: RelationKind::Inside,
                            },
                        ],
                    );
                }
                push(
                    "depart",
                    vec![
                        update(1, 20000, Pose::Seated),
                        update(2, 20000, Pose::Rest),
                        update(3, 20000, Pose::Rest),
                    ],
                    vec![
                        part,
                        Relation {
                            from: ObjectId(1),
                            to: ObjectId(2),
                            kind: RelationKind::Inside,
                        },
                    ],
                );
                phase = 5;
            }
            _ => {
                return Err(Error::Invalid(
                    "unsupported action or invalid taxi event order",
                ));
            }
        }
    }
    if phase != 5 {
        return Err(Error::Invalid("incomplete taxi scenario"));
    }
    Ok(Plan {
        prompt: prompt.to_owned(),
        world,
        events,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn taxi_replay_preserves_identity_and_containment() {
        let mut p = plan(TAXI_PROMPT).unwrap();
        assert_eq!(p, plan(TAXI_PROMPT).unwrap());
        assert_eq!(
            p.events.iter().map(|e| e.action).collect::<Vec<_>>(),
            [
                "hail",
                "approach",
                "open_door",
                "enter",
                "close_door",
                "depart"
            ]
        );
        let initial = p.world.current().objects().to_vec();
        for e in p.events {
            p.world
                .apply_scene(e.tick, &e.updates, Some(&e.relations), None)
                .unwrap();
        }
        for (old, new) in initial.iter().zip(p.world.current().objects()) {
            assert_eq!(
                (old.id, old.appearance, old.kind),
                (new.id, new.appearance, new.kind)
            );
        }
        let objects = p.world.current().objects();
        assert_eq!(objects[0].position, objects[1].position);
        assert_eq!(objects[0].pose, Pose::Seated);
        assert_eq!(objects[3], initial[3]);
        assert!(
            p.world
                .current()
                .relations()
                .iter()
                .any(|r| r.kind == RelationKind::Inside)
        );
    }
    #[test]
    fn control_scenario_has_same_six_ordered_state_transitions() {
        let detailed = plan(CONTROL_PROMPT).unwrap();
        let original = plan(TAXI_PROMPT).unwrap();
        assert_eq!(detailed.events, original.events);
        assert_eq!(detailed.world, original.world);
        assert_eq!(detailed.prompt, CONTROL_PROMPT);
        assert!(plan(&CONTROL_PROMPT.replace("жёлтое такси", "красный автобус")).is_err());
    }
    #[test]
    fn rejects_unknown_missing_reordered_and_oversized_text() {
        for text in [
            "",
            "A cat runs",
            "Человек стоит на улице, садится в машину. Такси уезжает",
            "Человек стоит на улице, останавливает такси",
            "Человек стоит на улице, летит на Марс",
        ] {
            assert!(plan(text).is_err());
        }
        assert!(plan(&"а".repeat(MAX_PROMPT_BYTES)).is_err());
        assert!(validate_prompt("hello\0world").is_err());
    }
}
