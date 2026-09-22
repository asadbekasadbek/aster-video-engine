//! Residency accounting. Backend callbacks must perform actual transfers; no tensor
//! movement or hardware telemetry is implied by these counters.
use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Ssd,
    Ram,
    Vram,
}
impl Tier {
    fn index(self) -> usize {
        match self {
            Self::Ssd => 0,
            Self::Ram => 1,
            Self::Vram => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resource {
    pub id: u32,
    pub bytes: usize,
    pub tier: Tier,
}

#[derive(Debug)]
pub struct Manager {
    resources: Vec<Resource>,
    limits: [usize; 3],
    used: [usize; 3],
    reserved_peak: [usize; 3],
}

impl Manager {
    pub fn new(limits: [usize; 3]) -> Self {
        Self {
            resources: Vec::new(),
            limits,
            used: [0; 3],
            reserved_peak: [0; 3],
        }
    }
    pub fn used(&self) -> [usize; 3] {
        self.used
    }
    pub fn reserved_peak(&self) -> [usize; 3] {
        self.reserved_peak
    }
    fn check(&self, tier: Tier, bytes: usize) -> Result<usize, Error> {
        let i = tier.index();
        let required = self.used[i].checked_add(bytes).ok_or(Error::Overflow)?;
        if required > self.limits[i] {
            return Err(Error::Budget {
                required,
                limit: self.limits[i],
            });
        }
        Ok(required)
    }
    /// Register an already resident resource; caller is responsible for truthful sizes.
    pub fn register(&mut self, resource: Resource) -> Result<(), Error> {
        if resource.bytes == 0
            || self.resources.len() >= 1024
            || self.resources.iter().any(|r| r.id == resource.id)
        {
            return Err(Error::Invalid("invalid, duplicate or excessive resource"));
        }
        let total = self.check(resource.tier, resource.bytes)?;
        self.resources
            .try_reserve(1)
            .map_err(|_| Error::Allocation)?;
        self.resources.push(resource);
        self.used[resource.tier.index()] = total;
        self.reserved_peak[resource.tier.index()] =
            self.reserved_peak[resource.tier.index()].max(total);
        Ok(())
    }
    /// Retain the source until a synchronous transfer succeeds. The callback must
    /// free its destination on failure; it must release the source on success.
    /// Temporary transfer workspace is charged to RAM while BOTH copies exist.
    pub fn transfer(
        &mut self,
        id: u32,
        destination: Tier,
        ram_workspace: usize,
        perform: impl FnOnce(Resource, Tier) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.transfer_value(id, destination, ram_workspace, perform)
    }

    /// Typed backend transfer, preserving the backend's error and returned value.
    /// Same overlap and rollback contract as `transfer`.
    pub fn transfer_value<T, E: From<Error>>(
        &mut self,
        id: u32,
        destination: Tier,
        ram_workspace: usize,
        perform: impl FnOnce(Resource, Tier) -> Result<T, E>,
    ) -> Result<T, E> {
        let index = self
            .resources
            .iter()
            .position(|r| r.id == id)
            .ok_or(Error::Invalid("unknown resource"))?;
        let resource = self.resources[index];
        if resource.tier.index().abs_diff(destination.index()) != 1 {
            return Err(
                Error::Invalid("transfer must move between adjacent residency tiers").into(),
            );
        }
        let mut overlap = self.used;
        overlap[destination.index()] = self.check(destination, resource.bytes)?;
        overlap[1] = overlap[1]
            .checked_add(ram_workspace)
            .ok_or(Error::Overflow)?;
        if overlap[1] > self.limits[1] {
            return Err(Error::Budget {
                required: overlap[1],
                limit: self.limits[1],
            }
            .into());
        }
        for (peak, reserved) in self.reserved_peak.iter_mut().zip(overlap) {
            *peak = (*peak).max(reserved);
        }
        let value = perform(resource, destination)?;
        self.used[resource.tier.index()] -= resource.bytes;
        self.used[destination.index()] += resource.bytes;
        self.resources[index].tier = destination;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_transfer_returns_value_and_preserves_backend_error() {
        let mut m = Manager::new([0, 4, 4]);
        m.register(Resource {
            id: 1,
            bytes: 4,
            tier: Tier::Ram,
        })
        .unwrap();
        let failed: Result<Vec<u8>, Box<dyn std::error::Error>> =
            m.transfer_value(1, Tier::Vram, 0, |_, _| {
                Err("backend transfer failed".into())
            });
        assert_eq!(failed.unwrap_err().to_string(), "backend transfer failed");
        assert_eq!(m.used(), [0, 4, 0]);
        let value: Result<Vec<u8>, Error> =
            m.transfer_value(1, Tier::Vram, 0, |_, _| Ok(vec![1, 2, 3, 4]));
        assert_eq!(value.unwrap(), [1, 2, 3, 4]);
        assert_eq!(m.used(), [0, 0, 4]);
    }
    #[test]
    fn staged_transfers_budget_overlap_and_rollback() {
        let mut m = Manager::new([1000, 110, 100]);
        m.register(Resource {
            id: 1,
            bytes: 100,
            tier: Tier::Ssd,
        })
        .unwrap();
        assert!(m.transfer(1, Tier::Vram, 0, |_, _| panic!()).is_err());
        assert!(m.transfer(1, Tier::Ram, 11, |_, _| panic!()).is_err());
        assert!(
            m.transfer(1, Tier::Ram, 10, |_, _| Err(Error::Cancelled))
                .is_err()
        );
        assert_eq!(m.used(), [100, 0, 0]);
        m.transfer(1, Tier::Ram, 10, |_, _| Ok(())).unwrap();
        m.transfer(1, Tier::Vram, 10, |_, _| Ok(())).unwrap();
        assert_eq!(m.used(), [0, 0, 100]);
        assert_eq!(m.reserved_peak(), [100, 110, 100]);
        m.transfer(1, Tier::Ram, 0, |_, _| Ok(())).unwrap();
        m.transfer(1, Tier::Ssd, 0, |_, _| Ok(())).unwrap();
        assert_eq!(m.used(), [100, 0, 0]);
    }
    #[test]
    fn overflow_duplicate_and_unknown_are_rejected() {
        let mut m = Manager::new([usize::MAX; 3]);
        m.register(Resource {
            id: 1,
            bytes: usize::MAX,
            tier: Tier::Ssd,
        })
        .unwrap();
        assert_eq!(
            m.register(Resource {
                id: 2,
                bytes: 1,
                tier: Tier::Ssd
            }),
            Err(Error::Overflow)
        );
        assert!(
            m.register(Resource {
                id: 1,
                bytes: 1,
                tier: Tier::Ram
            })
            .is_err()
        );
        assert!(m.transfer(99, Tier::Ram, 0, |_, _| panic!()).is_err());
    }
}
