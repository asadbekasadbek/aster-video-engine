use aster_core::{
    adaptive::{self, CacheKey, Capabilities, Mode},
    planner,
    world::ObjectId,
};

#[test]
fn text_to_world_to_changes_to_conservative_generation_policy() {
    let mut plan = planner::plan(planner::TAXI_PROMPT).unwrap();
    let mut affected = Vec::new();
    for event in plan.events {
        let before = plan.world.current().clone();
        plan.world
            .apply_scene(event.tick, &event.updates, Some(&event.relations), None)
            .unwrap();
        let changes = adaptive::detect(&before, plan.world.current(), &[], false);
        let key = CacheKey {
            model_revision: "unavailable".into(),
            conditioning: plan.prompt.clone(),
            frame: event.tick,
            seed: 42,
        };
        assert_eq!(
            adaptive::select(&changes, Capabilities::default(), None, &key).mode,
            Mode::Full
        );
        affected.push((event.action, changes.affected));
    }
    assert_eq!(affected[0], ("hail", vec![ObjectId(1)]));
    assert_eq!(affected[1], ("approach", vec![ObjectId(2), ObjectId(3)]));
    for (_, ids) in &affected[2..] {
        assert_eq!(ids, &[ObjectId(1), ObjectId(2), ObjectId(3)]);
    }
}
