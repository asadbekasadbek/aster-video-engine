//! Restricted symbolic-plan to text-conditioning boundary.
//! Text conditioning is not object grounding, masking, or guaranteed identity.
use crate::{Error, adaptive, planner};

#[derive(Debug)]
pub struct SceneConditioning {
    pub text: String,
    pub events: Vec<(u64, &'static str, adaptive::Changes, adaptive::Decision)>,
}

pub fn taxi(prompt: &str) -> Result<SceneConditioning, Error> {
    let mut plan = planner::plan(prompt)?;
    let mut text = String::from("A person stands on a city sidewalk. ");
    let mut events = Vec::new();
    for event in plan.events {
        let before = plan.world.current().clone();
        plan.world
            .apply_scene(event.tick, &event.updates, Some(&event.relations), None)?;
        let changes = adaptive::detect(&before, plan.world.current(), &[], false);
        let requested = adaptive::CacheKey {
            model_revision: "ltx-0.9.8-2b-distilled".into(),
            conditioning: prompt.into(),
            seed: 42,
            frame: event.tick,
        };
        let decision = adaptive::select(
            &changes,
            adaptive::Capabilities::default(),
            None,
            &requested,
        );
        text.push_str(match event.action {
            "hail" => "The person raises an arm to hail a taxi. ",
            "approach" => "The taxi approaches and stops beside the person. ",
            "open_door" => "The person opens the passenger door. ",
            "enter" => "The person gets inside and sits in the taxi. ",
            "close_door" => "The passenger door closes. ",
            "depart" => "The taxi drives away with the person inside. ",
            _ => return Err(Error::Invalid("unsupported neural scene action")),
        });
        events.push((event.tick, event.action, changes, decision));
    }
    text.push_str("Continuous shot, same person and taxi throughout.");
    Ok(SceneConditioning { text, events })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replay_drives_conditioning_and_conservative_decisions() {
        let result = taxi(planner::TAXI_PROMPT).unwrap();
        assert_eq!(result.events.len(), 6);
        assert!(
            result
                .events
                .iter()
                .all(|(_, _, changes, decision)| !changes.affected.is_empty()
                    && decision.mode == adaptive::Mode::Full)
        );
        assert!(result.text.find("opens").unwrap() < result.text.find("gets inside").unwrap());
        assert!(
            result.text.find("gets inside").unwrap() < result.text.find("drives away").unwrap()
        );
        assert!(taxi("A cat runs").is_err());
    }
}
