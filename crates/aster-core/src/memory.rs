//! OS memory counters, separate from tensor payload accounting.

#[cfg(windows)]
pub fn peak_working_set_bytes() -> std::io::Result<Option<usize>> {
    use std::ffi::c_void;
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_faults: u32,
        peak_working_set: usize,
        working_set: usize,
        quota_peak_paged: usize,
        quota_paged: usize,
        quota_peak_nonpaged: usize,
        quota_nonpaged: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(process: *mut c_void, counters: *mut Counters, cb: u32) -> i32;
    }
    let cb = size_of::<Counters>() as u32;
    let mut counters = Counters {
        cb,
        ..Counters::default()
    };
    // SAFETY: repr(C) matches PROCESS_MEMORY_COUNTERS; writable buffer has cb bytes.
    // The current-process pseudo handle is valid for this query and must not be closed.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, cb) };
    if ok == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(Some(counters.peak_working_set))
    }
}

#[cfg(not(windows))]
pub fn peak_working_set_bytes() -> std::io::Result<Option<usize>> {
    Ok(None)
}

#[cfg(all(test, windows))]
mod tests {
    #[test]
    fn reports_nonzero_process_peak() {
        assert!(super::peak_working_set_bytes().unwrap().unwrap() > 0);
    }
}
