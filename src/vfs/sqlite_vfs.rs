#![allow(unsafe_op_in_unsafe_fn)]

use std::ffi::{CStr, CString};
use std::mem::size_of;
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OpenFlags, ffi};
use thiserror::Error;

use crate::stego::device::{DeviceError, DeviceOptions, StegoDevice};
use crate::stego::planner::AllocationMode;

pub const SQLITE_VFS_NAME: &str = "llmdb-stego";
const SQLITE_HEADER_MAGIC: &[u8; 16] = b"SQLite format 3\0";
const SQLITE_PAGE_COUNT_OFFSET: usize = 28;
const SQLITE_PAGE_SIZE_OFFSET: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqliteVfsBootstrap {
    pub page_size: usize,
}

impl Default for SqliteVfsBootstrap {
    fn default() -> Self {
        Self {
            page_size: crate::BLOCK_SIZE,
        }
    }
}

pub fn initialize_model(
    path: impl AsRef<Path>,
    options: DeviceOptions,
) -> Result<(), SqliteVfsError> {
    let mut device = StegoDevice::initialize_with_options(path, AllocationMode::Standard, options)?;
    device.reserve_all_data_blocks()?;
    device.flush()?;
    Ok(())
}

pub fn open_connection(path: impl AsRef<Path>) -> Result<Connection, SqliteVfsError> {
    ensure_registered()?;

    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
        | OpenFlags::SQLITE_OPEN_CREATE
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let connection = Connection::open_with_flags_and_vfs(path, flags, SQLITE_VFS_NAME)?;
    configure_connection(&connection)?;
    Ok(connection)
}

fn configure_connection(connection: &Connection) -> Result<(), SqliteVfsError> {
    connection.execute_batch(&format!(
        "PRAGMA page_size={};
         PRAGMA temp_store=MEMORY;
         PRAGMA synchronous=FULL;
         PRAGMA wal_autocheckpoint=1;",
        crate::BLOCK_SIZE
    ))?;

    let mode: String = connection.query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(SqliteVfsError::UnexpectedJournalMode(mode));
    }

    Ok(())
}

fn ensure_registered() -> Result<(), SqliteVfsError> {
    static REGISTERED: OnceLock<c_int> = OnceLock::new();
    let rc = *REGISTERED.get_or_init(|| unsafe {
        let rc = ffi::sqlite3_initialize();
        if rc != ffi::SQLITE_OK {
            return rc;
        }

        let name = CString::new(SQLITE_VFS_NAME).expect("static vfs name");
        let leaked_name = Box::leak(name.into_boxed_c_str());
        let vfs = Box::leak(Box::new(ffi::sqlite3_vfs {
            iVersion: 1,
            szOsFile: size_of::<LlmdbFile>() as c_int,
            mxPathname: 1024,
            pNext: ptr::null_mut(),
            zName: leaked_name.as_ptr(),
            pAppData: ptr::null_mut(),
            xOpen: Some(vfs_open),
            xDelete: Some(vfs_delete),
            xAccess: Some(vfs_access),
            xFullPathname: Some(vfs_full_pathname),
            xDlOpen: Some(vfs_dlopen),
            xDlError: Some(vfs_dlerror),
            xDlSym: Some(vfs_dlsym),
            xDlClose: Some(vfs_dlclose),
            xRandomness: Some(vfs_randomness),
            xSleep: Some(vfs_sleep),
            xCurrentTime: Some(vfs_current_time),
            xGetLastError: Some(vfs_get_last_error),
            xCurrentTimeInt64: None,
            xSetSystemCall: None,
            xGetSystemCall: None,
            xNextSystemCall: None,
        }));

        ffi::sqlite3_vfs_register(vfs, 0)
    });

    if rc == ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(SqliteVfsError::Register(rc))
    }
}

#[repr(C)]
struct LlmdbFile {
    base: ffi::sqlite3_file,
    lock_level: c_int,
    backend: FileBackend,
}

impl LlmdbFile {
    fn new(backend: FileBackend) -> Self {
        Self {
            base: ffi::sqlite3_file {
                pMethods: &IO_METHODS,
            },
            lock_level: ffi::SQLITE_LOCK_NONE,
            backend,
        }
    }
}

enum FileBackend {
    Main(MainFileState),
    Memory(MemoryFileState),
}

struct MainFileState {
    device: StegoDevice,
    logical_size: i64,
    shm_regions: Vec<Box<[u8]>>,
}

struct MemoryFileState {
    bytes: Vec<u8>,
}

static IO_METHODS: ffi::sqlite3_io_methods = ffi::sqlite3_io_methods {
    iVersion: 3,
    xClose: Some(file_close),
    xRead: Some(file_read),
    xWrite: Some(file_write),
    xTruncate: Some(file_truncate),
    xSync: Some(file_sync),
    xFileSize: Some(file_size),
    xLock: Some(file_lock),
    xUnlock: Some(file_unlock),
    xCheckReservedLock: Some(file_check_reserved_lock),
    xFileControl: Some(file_control),
    xSectorSize: Some(file_sector_size),
    xDeviceCharacteristics: Some(file_device_characteristics),
    xShmMap: Some(file_shm_map),
    xShmLock: Some(file_shm_lock),
    xShmBarrier: Some(file_shm_barrier),
    xShmUnmap: Some(file_shm_unmap),
    xFetch: Some(file_fetch),
    xUnfetch: Some(file_unfetch),
};

unsafe extern "C" fn vfs_open(
    _vfs: *mut ffi::sqlite3_vfs,
    z_name: ffi::sqlite3_filename,
    file: *mut ffi::sqlite3_file,
    flags: c_int,
    out_flags: *mut c_int,
) -> c_int {
    if !out_flags.is_null() {
        *out_flags = flags;
    }

    let path = filename_to_string(z_name);
    let backend = if flags & ffi::SQLITE_OPEN_MAIN_DB != 0 {
        match open_main_backend(&path) {
            Ok(backend) => FileBackend::Main(backend),
            Err(error) => return device_error_code(&error),
        }
    } else {
        FileBackend::Memory(MemoryFileState { bytes: Vec::new() })
    };

    ptr::write(file.cast::<LlmdbFile>(), LlmdbFile::new(backend));
    ffi::SQLITE_OK
}

unsafe extern "C" fn vfs_delete(
    _vfs: *mut ffi::sqlite3_vfs,
    z_name: *const c_char,
    _sync_dir: c_int,
) -> c_int {
    let path = filename_to_string(z_name);
    if is_auxiliary_sqlite_path(&path) {
        ffi::SQLITE_OK
    } else {
        ffi::SQLITE_IOERR_DELETE
    }
}

unsafe extern "C" fn vfs_access(
    _vfs: *mut ffi::sqlite3_vfs,
    z_name: *const c_char,
    flags: c_int,
    out_res: *mut c_int,
) -> c_int {
    let path = filename_to_string(z_name);
    let exists = if is_auxiliary_sqlite_path(&path) {
        0
    } else {
        i32::from(Path::new(&path).exists())
    };

    if !out_res.is_null() {
        *out_res = match flags {
            ffi::SQLITE_ACCESS_EXISTS | ffi::SQLITE_ACCESS_READWRITE | ffi::SQLITE_ACCESS_READ => {
                exists
            }
            _ => exists,
        };
    }

    ffi::SQLITE_OK
}

unsafe extern "C" fn vfs_full_pathname(
    _vfs: *mut ffi::sqlite3_vfs,
    z_name: *const c_char,
    n_out: c_int,
    z_out: *mut c_char,
) -> c_int {
    if z_out.is_null() || n_out <= 0 {
        return ffi::SQLITE_IOERR;
    }

    let name = filename_to_string(z_name);
    let bytes = name.as_bytes();
    let max_len = (n_out as usize).saturating_sub(1);
    let copied = bytes.len().min(max_len);
    ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), z_out, copied);
    *z_out.add(copied) = 0;
    ffi::SQLITE_OK
}

unsafe extern "C" fn vfs_dlopen(
    _vfs: *mut ffi::sqlite3_vfs,
    _z_filename: *const c_char,
) -> *mut c_void {
    ptr::null_mut()
}

unsafe extern "C" fn vfs_dlerror(
    _vfs: *mut ffi::sqlite3_vfs,
    n_byte: c_int,
    z_err_msg: *mut c_char,
) {
    if z_err_msg.is_null() || n_byte <= 0 {
        return;
    }

    let message = b"loadable extensions are disabled\0";
    let len = message.len().min(n_byte as usize);
    ptr::copy_nonoverlapping(message.as_ptr().cast::<c_char>(), z_err_msg, len);
}

unsafe extern "C" fn vfs_dlsym(
    _vfs: *mut ffi::sqlite3_vfs,
    _handle: *mut c_void,
    _z_symbol: *const c_char,
) -> Option<unsafe extern "C" fn(*mut ffi::sqlite3_vfs, *mut c_void, *const c_char)> {
    None
}

unsafe extern "C" fn vfs_dlclose(_vfs: *mut ffi::sqlite3_vfs, _handle: *mut c_void) {}

unsafe extern "C" fn vfs_randomness(
    _vfs: *mut ffi::sqlite3_vfs,
    n_byte: c_int,
    z_out: *mut c_char,
) -> c_int {
    if z_out.is_null() || n_byte <= 0 {
        return 0;
    }

    let out = slice::from_raw_parts_mut(z_out.cast::<u8>(), n_byte as usize);
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_le_bytes();
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = seed[index % seed.len()].wrapping_add(index as u8);
    }
    n_byte
}

unsafe extern "C" fn vfs_sleep(_vfs: *mut ffi::sqlite3_vfs, microseconds: c_int) -> c_int {
    if microseconds > 0 {
        thread::sleep(Duration::from_micros(microseconds as u64));
    }
    microseconds
}

unsafe extern "C" fn vfs_current_time(_vfs: *mut ffi::sqlite3_vfs, out: *mut f64) -> c_int {
    if out.is_null() {
        return ffi::SQLITE_IOERR;
    }

    let unix_days = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
        / 86_400.0;
    *out = unix_days + 2_440_587.5;
    ffi::SQLITE_OK
}

unsafe extern "C" fn vfs_get_last_error(
    _vfs: *mut ffi::sqlite3_vfs,
    _n_byte: c_int,
    _z_err_msg: *mut c_char,
) -> c_int {
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_close(file: *mut ffi::sqlite3_file) -> c_int {
    ptr::drop_in_place(file.cast::<LlmdbFile>());
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_read(
    file: *mut ffi::sqlite3_file,
    out: *mut c_void,
    amount: c_int,
    offset: ffi::sqlite3_int64,
) -> c_int {
    let llmdb = &mut *file.cast::<LlmdbFile>();
    let buffer = slice::from_raw_parts_mut(out.cast::<u8>(), amount as usize);

    match &mut llmdb.backend {
        FileBackend::Main(state) => read_main(state, buffer, offset),
        FileBackend::Memory(state) => read_memory(state, buffer, offset),
    }
}

unsafe extern "C" fn file_write(
    file: *mut ffi::sqlite3_file,
    data: *const c_void,
    amount: c_int,
    offset: ffi::sqlite3_int64,
) -> c_int {
    let llmdb = &mut *file.cast::<LlmdbFile>();
    let buffer = slice::from_raw_parts(data.cast::<u8>(), amount as usize);

    match &mut llmdb.backend {
        FileBackend::Main(state) => write_main(state, buffer, offset),
        FileBackend::Memory(state) => write_memory(state, buffer, offset),
    }
}

unsafe extern "C" fn file_truncate(
    file: *mut ffi::sqlite3_file,
    size: ffi::sqlite3_int64,
) -> c_int {
    let llmdb = &mut *file.cast::<LlmdbFile>();
    match &mut llmdb.backend {
        FileBackend::Main(state) => {
            state.logical_size = size.max(0);
            ffi::SQLITE_OK
        }
        FileBackend::Memory(state) => {
            let new_len = size.max(0) as usize;
            state.bytes.resize(new_len, 0);
            ffi::SQLITE_OK
        }
    }
}

unsafe extern "C" fn file_sync(file: *mut ffi::sqlite3_file, _flags: c_int) -> c_int {
    let llmdb = &mut *file.cast::<LlmdbFile>();
    match &mut llmdb.backend {
        FileBackend::Main(state) => match state.device.flush() {
            Ok(()) => ffi::SQLITE_OK,
            Err(error) => device_error_code(&error),
        },
        FileBackend::Memory(_) => ffi::SQLITE_OK,
    }
}

unsafe extern "C" fn file_size(
    file: *mut ffi::sqlite3_file,
    out_size: *mut ffi::sqlite3_int64,
) -> c_int {
    if out_size.is_null() {
        return ffi::SQLITE_IOERR;
    }

    let llmdb = &mut *file.cast::<LlmdbFile>();
    *out_size = match &llmdb.backend {
        FileBackend::Main(state) => state.logical_size,
        FileBackend::Memory(state) => state.bytes.len() as i64,
    };
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_lock(file: *mut ffi::sqlite3_file, level: c_int) -> c_int {
    let llmdb = &mut *file.cast::<LlmdbFile>();
    llmdb.lock_level = llmdb.lock_level.max(level);
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_unlock(file: *mut ffi::sqlite3_file, level: c_int) -> c_int {
    let llmdb = &mut *file.cast::<LlmdbFile>();
    llmdb.lock_level = level;
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_check_reserved_lock(
    file: *mut ffi::sqlite3_file,
    out_res: *mut c_int,
) -> c_int {
    if out_res.is_null() {
        return ffi::SQLITE_IOERR;
    }

    let llmdb = &mut *file.cast::<LlmdbFile>();
    *out_res = i32::from(llmdb.lock_level >= ffi::SQLITE_LOCK_RESERVED);
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_control(
    _file: *mut ffi::sqlite3_file,
    _op: c_int,
    _arg: *mut c_void,
) -> c_int {
    ffi::SQLITE_NOTFOUND
}

unsafe extern "C" fn file_sector_size(_file: *mut ffi::sqlite3_file) -> c_int {
    crate::BLOCK_SIZE as c_int
}

unsafe extern "C" fn file_device_characteristics(_file: *mut ffi::sqlite3_file) -> c_int {
    ffi::SQLITE_IOCAP_ATOMIC4K
        | ffi::SQLITE_IOCAP_SAFE_APPEND
        | ffi::SQLITE_IOCAP_POWERSAFE_OVERWRITE
}

unsafe extern "C" fn file_shm_map(
    file: *mut ffi::sqlite3_file,
    page_index: c_int,
    page_size: c_int,
    extend: c_int,
    out_region: *mut *mut c_void,
) -> c_int {
    if out_region.is_null() {
        return ffi::SQLITE_IOERR;
    }

    let llmdb = &mut *file.cast::<LlmdbFile>();
    let FileBackend::Main(state) = &mut llmdb.backend else {
        return ffi::SQLITE_IOERR;
    };

    let page_index = page_index.max(0) as usize;
    while state.shm_regions.len() <= page_index {
        if extend == 0 {
            *out_region = ptr::null_mut();
            return ffi::SQLITE_OK;
        }
        state
            .shm_regions
            .push(vec![0_u8; page_size as usize].into_boxed_slice());
    }

    *out_region = state.shm_regions[page_index].as_mut_ptr().cast::<c_void>();
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_shm_lock(
    _file: *mut ffi::sqlite3_file,
    _offset: c_int,
    _n: c_int,
    _flags: c_int,
) -> c_int {
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_shm_barrier(_file: *mut ffi::sqlite3_file) {}

unsafe extern "C" fn file_shm_unmap(file: *mut ffi::sqlite3_file, delete_flag: c_int) -> c_int {
    let llmdb = &mut *file.cast::<LlmdbFile>();
    if delete_flag != 0 {
        if let FileBackend::Main(state) = &mut llmdb.backend {
            state.shm_regions.clear();
        }
    }
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_fetch(
    _file: *mut ffi::sqlite3_file,
    _offset: ffi::sqlite3_int64,
    _amount: c_int,
    out: *mut *mut c_void,
) -> c_int {
    if !out.is_null() {
        *out = ptr::null_mut();
    }
    ffi::SQLITE_OK
}

unsafe extern "C" fn file_unfetch(
    _file: *mut ffi::sqlite3_file,
    _offset: ffi::sqlite3_int64,
    _ptr: *mut c_void,
) -> c_int {
    ffi::SQLITE_OK
}

fn open_main_backend(path: &str) -> Result<MainFileState, DeviceError> {
    let device = StegoDevice::open_with_options(
        path,
        AllocationMode::Standard,
        DeviceOptions { verbose: false },
    )?;
    let logical_size = detect_sqlite_file_size(&device)?;

    Ok(MainFileState {
        device,
        logical_size,
        shm_regions: Vec::new(),
    })
}

fn detect_sqlite_file_size(device: &StegoDevice) -> Result<i64, DeviceError> {
    let first_block = device.data_region_start();
    if first_block >= device.shadow_block() {
        return Ok(0);
    }

    let page = device.read_block(first_block)?;
    if &page[..SQLITE_HEADER_MAGIC.len()] != SQLITE_HEADER_MAGIC {
        return Ok(0);
    }

    let page_size = u16::from_be_bytes([
        page[SQLITE_PAGE_SIZE_OFFSET],
        page[SQLITE_PAGE_SIZE_OFFSET + 1],
    ]);
    let page_size = if page_size == 1 {
        65_536
    } else {
        page_size as u32
    };
    if page_size as usize != crate::BLOCK_SIZE {
        return Ok(0);
    }

    let page_count = u32::from_be_bytes(
        page[SQLITE_PAGE_COUNT_OFFSET..SQLITE_PAGE_COUNT_OFFSET + 4]
            .try_into()
            .unwrap(),
    );

    Ok(i64::from(page_count.max(1)) * page_size as i64)
}

fn read_main(state: &mut MainFileState, out: &mut [u8], offset: i64) -> c_int {
    if offset < 0 {
        return ffi::SQLITE_IOERR;
    }

    out.fill(0);
    let file_size = state.logical_size.max(0);
    if offset >= file_size {
        return ffi::SQLITE_IOERR_SHORT_READ;
    }

    let available = ((file_size - offset) as usize).min(out.len());
    let end = offset as usize + available;
    let mut copied = 0_usize;
    let start = offset as usize;

    while start + copied < end {
        let absolute = start + copied;
        let page_index = absolute / crate::BLOCK_SIZE;
        let in_page = absolute % crate::BLOCK_SIZE;
        let block_index = state.device.data_region_start() + page_index as u32;
        if block_index >= state.device.shadow_block() {
            return ffi::SQLITE_IOERR;
        }

        let page = match state.device.read_block(block_index) {
            Ok(page) => page,
            Err(error) => return device_error_code(&error),
        };

        let chunk = (crate::BLOCK_SIZE - in_page).min(available - copied);
        out[copied..copied + chunk].copy_from_slice(&page[in_page..in_page + chunk]);
        copied += chunk;
    }

    if available < out.len() {
        ffi::SQLITE_IOERR_SHORT_READ
    } else {
        ffi::SQLITE_OK
    }
}

fn write_main(state: &mut MainFileState, data: &[u8], offset: i64) -> c_int {
    if offset < 0 {
        return ffi::SQLITE_IOERR;
    }

    let end = offset as usize + data.len();
    let max_bytes = (state.device.shadow_block() - state.device.data_region_start()) as usize
        * crate::BLOCK_SIZE;
    if end > max_bytes {
        return ffi::SQLITE_FULL;
    }

    let mut written = 0_usize;
    let start = offset as usize;
    while written < data.len() {
        let absolute = start + written;
        let page_index = absolute / crate::BLOCK_SIZE;
        let in_page = absolute % crate::BLOCK_SIZE;
        let block_index = state.device.data_region_start() + page_index as u32;

        let page_start = page_index * crate::BLOCK_SIZE;
        let mut page = if (page_start as i64) < state.logical_size {
            match state.device.read_block(block_index) {
                Ok(page) => page,
                Err(error) => return device_error_code(&error),
            }
        } else {
            vec![0_u8; crate::BLOCK_SIZE]
        };

        let chunk = (crate::BLOCK_SIZE - in_page).min(data.len() - written);
        page[in_page..in_page + chunk].copy_from_slice(&data[written..written + chunk]);

        if let Err(error) = state.device.write_block(block_index, &page) {
            return device_error_code(&error);
        }

        written += chunk;
    }

    state.logical_size = state.logical_size.max(offset + data.len() as i64);
    ffi::SQLITE_OK
}

fn read_memory(state: &mut MemoryFileState, out: &mut [u8], offset: i64) -> c_int {
    if offset < 0 {
        return ffi::SQLITE_IOERR;
    }

    out.fill(0);
    let offset = offset as usize;
    if offset >= state.bytes.len() {
        return ffi::SQLITE_IOERR_SHORT_READ;
    }

    let available = (state.bytes.len() - offset).min(out.len());
    out[..available].copy_from_slice(&state.bytes[offset..offset + available]);

    if available < out.len() {
        ffi::SQLITE_IOERR_SHORT_READ
    } else {
        ffi::SQLITE_OK
    }
}

fn write_memory(state: &mut MemoryFileState, data: &[u8], offset: i64) -> c_int {
    if offset < 0 {
        return ffi::SQLITE_IOERR;
    }

    let offset = offset as usize;
    let end = offset + data.len();
    if state.bytes.len() < end {
        state.bytes.resize(end, 0);
    }
    state.bytes[offset..end].copy_from_slice(data);
    ffi::SQLITE_OK
}

fn filename_to_string(z_name: *const c_char) -> String {
    if z_name.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(z_name) }
            .to_string_lossy()
            .into_owned()
    }
}

fn is_auxiliary_sqlite_path(path: &str) -> bool {
    path.ends_with("-wal")
        || path.ends_with("-journal")
        || path.ends_with("-shm")
        || path.ends_with("-sj")
}

fn device_error_code(error: &DeviceError) -> c_int {
    match error {
        DeviceError::OutOfSpace => ffi::SQLITE_FULL,
        DeviceError::IntegrityMismatch { .. } | DeviceError::PendingWriteCrcMismatch { .. } => {
            ffi::SQLITE_CORRUPT
        }
        _ => ffi::SQLITE_IOERR,
    }
}

#[derive(Debug, Error)]
pub enum SqliteVfsError {
    #[error("device error: {0}")]
    Device(#[from] DeviceError),
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("failed to register sqlite vfs: rc={0}")]
    Register(c_int),
    #[error("expected WAL mode, got {0}")]
    UnexpectedJournalMode(String),
}
