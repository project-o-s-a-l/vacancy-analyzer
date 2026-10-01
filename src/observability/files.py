"""Bounded asynchronous JSONL storage. Only managed names inside one private directory."""

import errno
import os
import queue
import re
import stat
import threading
import time
from pathlib import Path

NAME = re.compile(
    r"^(va-(?:cli|bot|collect|worker|semantic_worker|web)-[a-f0-9]{32})\.jsonl(?:\.([1-9][0-9]?))?$"
)
MAX_ENTRIES = 4096
MAX_LINE_BYTES = 4096
RETRY_DELAY = 1
MAINTENANCE_INTERVAL = 60


class StorageFailure(Exception):
    def __init__(self, code):
        self.code = code


def failure_code(exc):
    if isinstance(exc, StorageFailure):
        return exc.code
    if isinstance(exc, OSError):
        return {
            errno.ENOSPC: "log_disk_full",
            errno.EACCES: "log_permission_denied",
            errno.EPERM: "log_permission_denied",
        }.get(exc.errno, "log_io_error")
    return "log_io_error"


def plain(path, *, directory=False):
    """Do not follow symlinks, Windows junctions or hard links to outside data."""
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & 0x400:
        raise StorageFailure("log_unsafe_path")
    if directory:
        if not stat.S_ISDIR(value.st_mode):
            raise StorageFailure("log_unsafe_path")
    elif not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise StorageFailure("log_unsafe_path")
    return True


def open_private(path, flags):
    plain(path)
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0), 0o600)
    try:
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise StorageFailure("log_unsafe_path")
    except BaseException:
        os.close(fd)
        raise
    return fd


def lock(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise
        return False


def lock_file(path):
    fd = open_private(path, os.O_CREAT | os.O_RDWR)
    if os.fstat(fd).st_size == 0:
        os.write(fd, b"0")
    return fd


class Store:
    def __init__(self, options, role, run_id):
        self.options = options
        self.root = Path(os.path.abspath(options.directory))
        self.stem = f"va-{role}-{run_id.replace('-', '')}"
        self.path = self.root / (self.stem + ".jsonl")
        self.lease = None
        self.identity = None

    def prepare(self):
        if self.root == self.root.parent or self.root == Path.cwd():
            raise StorageFailure("log_unsafe_path")
        for parent in reversed((self.root, *self.root.parents)):
            plain(parent, directory=True)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        plain(self.root, directory=True)
        value = self.root.stat()
        identity = value.st_dev, value.st_ino
        if self.identity is not None and identity != self.identity:
            raise StorageFailure("log_unsafe_path")
        self.identity = identity

    def _global_lock(self):
        self.prepare()
        fd = lock_file(self.root / ".va-log-budget.lock")
        deadline = time.monotonic() + 0.5
        try:
            while not lock(fd):
                if time.monotonic() >= deadline:
                    raise StorageFailure("log_lock_busy")
                time.sleep(0.005)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _groups(self):
        groups, leases = {}, {}
        with os.scandir(self.root) as entries:
            for index, entry in enumerate(entries):
                if index >= MAX_ENTRIES:
                    raise StorageFailure("log_directory_limit")
                match = NAME.fullmatch(entry.name)
                lease_match = (
                    NAME.fullmatch(entry.name[:-6] + ".jsonl")
                    if entry.name.endswith(".lease")
                    else None
                )
                if not match and not lease_match:
                    continue
                path = self.root / entry.name
                plain(path)
                if lease_match:
                    leases[lease_match[1]] = path
                    continue
                value = path.stat()
                group = groups.setdefault(match[1], {"files": [], "size": 0, "newest": 0})
                group["files"].append(path)
                group["size"] += value.st_size
                group["newest"] = max(group["newest"], value.st_mtime)
        return groups, leases

    def _inactive(self, stem, leases):
        if stem == self.stem:
            return False
        path = leases.get(stem)
        if path is None:
            return True
        fd = open_private(path, os.O_RDWR)
        try:
            return lock(fd)  # Closing the descriptor releases the acquired test lock.
        finally:
            os.close(fd)

    def _delete_group(self, group, lease):
        for path in group["files"]:
            plain(path)
            path.unlink()
        if lease:
            plain(lease)
            lease.unlink()

    def _cleanup(self, extra):
        groups, leases = self._groups()
        total = sum(g["size"] for g in groups.values())
        cutoff = time.time() - self.options.retention_days * 86400
        closed = [(stem, g) for stem, g in groups.items() if self._inactive(stem, leases)]
        for stem, group in sorted(closed, key=lambda item: item[1]["newest"]):
            if group["newest"] < cutoff or total + extra > self.options.total_max_bytes:
                self._delete_group(group, leases.get(stem))
                total -= group["size"]
        # A process can fail before creating its first JSONL record.
        for stem, path in leases.items():
            if stem not in groups and self._inactive(stem, leases):
                plain(path)
                path.unlink()
        if total + extra > self.options.total_max_bytes:
            raise StorageFailure("log_budget_exceeded")

    def _rotate(self):
        try:
            count = self.options.backup_count
            oldest = self.root / (self.path.name + f".{count}") if count else self.path
            if plain(oldest):
                oldest.unlink()
            for index in range(count - 1, 0, -1):
                old = self.root / (self.path.name + f".{index}")
                new = self.root / (self.path.name + f".{index + 1}")
                if plain(old):
                    plain(new)
                    old.replace(new)
            if count and plain(self.path):
                target = self.root / (self.path.name + ".1")
                plain(target)
                self.path.replace(target)
        except OSError as exc:
            raise StorageFailure("log_rotation_failed") from exc

    def write(self, data):
        if type(data) is not bytes or not 0 < len(data) <= min(
            MAX_LINE_BYTES, self.options.max_file_bytes
        ):
            raise StorageFailure("log_record_invalid")
        fd = self._global_lock()
        try:
            if self.lease is None:
                lease = lock_file(self.root / (self.stem + ".lease"))
                if not lock(lease):
                    os.close(lease)
                    raise StorageFailure("log_lock_busy")
                self.lease = lease
            else:
                lease_path = self.root / (self.stem + ".lease")
                if not plain(lease_path):
                    raise StorageFailure("log_unsafe_path")
                current, opened = lease_path.stat(), os.fstat(self.lease)
                if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                    raise StorageFailure("log_unsafe_path")
            if (
                plain(self.path)
                and self.path.stat().st_size + len(data) > self.options.max_file_bytes
            ):
                self._rotate()
            self._cleanup(len(data))
            target = open_private(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY)
            before = os.fstat(target).st_size
            try:
                remaining = memoryview(data)
                while remaining:
                    written = os.write(target, remaining)
                    if not written:
                        raise OSError(errno.ENOSPC, "")
                    remaining = remaining[written:]
            except OSError:
                try:
                    os.ftruncate(target, before)
                except OSError:
                    pass  # A failing disk cannot guarantee a complete final record.
                raise
            finally:
                os.close(target)
        finally:
            os.close(fd)

    def close(self):
        if self.lease is not None:
            os.close(self.lease)
            self.lease = None

    def maintenance(self):
        fd = self._global_lock()
        try:
            self._cleanup(0)
        finally:
            os.close(fd)


class FileJournal:
    """The producer queues <=4096-byte already-sanitized lines; only this thread does I/O."""

    def __init__(self, options, role, run_id, notice):
        self.options = options
        self.store = Store(options, role, run_id)
        self.queue = queue.Queue(maxsize=options.queue_size)
        self.stop = threading.Event()
        self.notice = notice
        self.mutex = threading.Lock()
        self.dropped = 0
        self.inflight = False
        self.last_code = None
        self.last_notice = float("-inf")
        self.thread = threading.Thread(target=self._run, name="application-log-writer", daemon=True)
        self.thread.start()

    def _lost(self, count=1):
        with self.mutex:
            self.dropped = min(10**9, self.dropped + count)

    def _notify(self, code, *, force=False, pending_count=0):
        now = time.monotonic()
        with self.mutex:
            if (
                not force
                and code == self.last_code
                and now - self.last_notice < self.options.repeat_interval
            ):
                return
            self.last_code, self.last_notice = code, now
            count = self.dropped
        self.notice(code, count, pending_count)

    def enqueue(self, data):
        if type(data) is not bytes or not 0 < len(data) <= MAX_LINE_BYTES:
            self._lost()
            self._notify("log_record_invalid")
            return
        if self.stop.is_set():
            self._lost()
            return
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            self._lost()
            self._notify("log_queue_full")

    def _run(self):
        next_attempt = 0
        next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    data = self.queue.get(timeout=0.05)
                except queue.Empty:
                    if self.store.lease is not None and time.monotonic() >= next_maintenance:
                        next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL
                        try:
                            self.store.maintenance()
                        except Exception as exc:
                            self._notify(failure_code(exc))
                    continue
                try:
                    self.inflight = True
                    if time.monotonic() < next_attempt:
                        self._lost()
                        self._notify(self.last_code or "log_io_error")
                        continue
                    try:
                        self.store.write(data)
                    except Exception as exc:
                        self._lost()
                        self._notify(failure_code(exc))
                        next_attempt = time.monotonic() + RETRY_DELAY
                    else:
                        if self.last_code is not None and self.queue.empty():
                            self._notify("log_recovered")
                            with self.mutex:
                                self.last_code = None
                finally:
                    self.inflight = False
                    self.queue.task_done()
        finally:
            try:
                self.store.close()
            except OSError:
                self._notify("log_io_error")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=self.options.shutdown_timeout)
        if self.thread.is_alive():
            self._notify(
                "log_shutdown_timeout",
                force=True,
                pending_count=self.queue.qsize() + int(self.inflight),
            )
        elif self.dropped:
            self._notify(self.last_code or "log_records_lost", force=True)
