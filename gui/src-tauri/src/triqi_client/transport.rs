//! In-memory full-duplex byte transport.
//!
//! Lets the gui run the `triq serve` loop in-process (one binary,
//! no child): [`duplex`] produces two cross-connected ends — bytes written to
//! one end's [`DuplexWriter`] become readable from the other end's
//! [`DuplexReader`], in both directions. Both halves implement
//! `std::io::Read`/`std::io::Write`, so a protocol loop written for stdio
//! pipes runs unchanged over it.
//!
//! Semantics mirror the stdio pipes the design relies on (`gui/design/app1/thin-gui.md`):
//! - writes never block (the buffer is unbounded) — the same invariant as the
//!   dedicated reader/writer tasks, which always drain;
//! - dropping a `DuplexWriter` is EOF for the peer reader: reads return
//!   `Ok(0)` once the buffer is drained, exactly like a closed pipe.

use std::collections::VecDeque;
use std::io::{BufRead, Read, Write};
use std::sync::{Arc, Condvar, Mutex};

/// Shared buffer for one direction of a [`duplex`] pair.
struct Shared {
    buf: Mutex<VecDeque<u8>>,
    /// Notified on push and on close.
    cv: Condvar,
    /// Set when the writing half was dropped: drained buffer = EOF.
    closed: Mutex<bool>,
}

/// Writing half of one direction. Dropping it signals EOF to the peer reader.
pub struct DuplexWriter {
    shared: Arc<Shared>,
}

/// Reading half of one direction.
pub struct DuplexReader {
    shared: Arc<Shared>,
    /// Bytes drained from the shared buffer but not yet handed to the caller
    /// (`fill_buf` must return a contiguous slice that outlives the lock).
    scratch: Vec<u8>,
    scratch_pos: usize,
}

/// A full-duplex in-memory channel: `(a_write, a_read, b_write, b_read)`.
/// Bytes written to `a_write` are read from `b_read`, and vice versa.
///
/// Side ownership matters: side A owns `a_write` + `a_read` (it writes
/// requests with `a_write` and reads the responses arriving on `a_read`),
/// side B owns `b_write` + `b_read`. Mixing readers across sides silently
/// wires a peer to its own traffic — pair each side's writer with ITS OWN
/// reader (see `round_trip_both_directions` and the `over` constructors).
pub fn duplex() -> (DuplexWriter, DuplexReader, DuplexWriter, DuplexReader) {
    let a2b = Arc::new(Shared {
        buf: Mutex::new(VecDeque::new()),
        cv: Condvar::new(),
        closed: Mutex::new(false),
    });
    let b2a = Arc::new(Shared {
        buf: Mutex::new(VecDeque::new()),
        cv: Condvar::new(),
        closed: Mutex::new(false),
    });
    (
        DuplexWriter { shared: Arc::clone(&a2b) },
        DuplexReader {
            shared: Arc::clone(&b2a),
            scratch: Vec::new(),
            scratch_pos: 0,
        },
        DuplexWriter { shared: Arc::clone(&b2a) },
        DuplexReader {
            shared: Arc::clone(&a2b),
            scratch: Vec::new(),
            scratch_pos: 0,
        },
    )
}

impl Write for DuplexWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut q = self.shared.buf.lock().expect("duplex buf lock");
        q.extend(buf.iter().copied());
        drop(q);
        self.shared.cv.notify_all();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl DuplexReader {
    /// Block until data is available (or EOF), then move up to one chunk into
    /// the scratch buffer for [`BufRead::fill_buf`] to hand out.
    fn refill_scratch(&mut self) -> std::io::Result<()> {
        if self.scratch_pos < self.scratch.len() {
            return Ok(());
        }
        self.scratch.clear();
        self.scratch_pos = 0;
        let mut q = self.shared.buf.lock().expect("duplex buf lock");
        loop {
            if !q.is_empty() {
                let take = q.len().min(64 * 1024);
                self.scratch.extend(q.drain(..take));
                return Ok(());
            }
            let closed = *self.shared.closed.lock().expect("duplex closed lock");
            if closed {
                return Ok(()); // EOF: writer gone, buffer drained
            }
            q = self.shared.cv.wait(q).expect("duplex cv wait");
        }
    }
}

impl Read for DuplexReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            self.refill_scratch()?;
            if self.scratch_pos < self.scratch.len() {
                let n = out.len().min(self.scratch.len() - self.scratch_pos);
                out[..n].copy_from_slice(&self.scratch[self.scratch_pos..self.scratch_pos + n]);
                self.scratch_pos += n;
                return Ok(n);
            }
            // EOF (refill found an empty, closed buffer).
            return Ok(0);
        }
    }
}

impl BufRead for DuplexReader {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.refill_scratch()?;
        Ok(&self.scratch[self.scratch_pos..])
    }

    fn consume(&mut self, amt: usize) {
        self.scratch_pos = self.scratch_pos.saturating_add(amt);
    }
}

impl Drop for DuplexWriter {
    fn drop(&mut self) {
        *self.shared.closed.lock().expect("duplex closed lock") = true;
        self.shared.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn round_trip_both_directions() {
        let (mut aw, mut ar, mut bw, mut br) = duplex();
        aw.write_all(b"ping\n").unwrap();
        bw.write_all(b"pong\n").unwrap();
        let mut buf = [0u8; 5];
        br.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ping\n");
        ar.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"pong\n");
    }

    #[test]
    fn partial_reads_accumulate() {
        let (mut aw, _ar, _bw, mut br) = duplex();
        aw.write_all(b"hello").unwrap();
        let mut one = [0u8; 2];
        br.read_exact(&mut one).unwrap();
        assert_eq!(&one, b"he");
        // read_to_string blocks until EOF; drop the writer first.
        drop(aw);
        let mut s = String::new();
        use std::io::Read as _;
        br.read_to_string(&mut s).unwrap();
        assert_eq!(s, "llo");
    }

    #[test]
    fn writer_drop_is_eof_for_peer() {
        let (mut aw, _ar, _bw, mut br) = duplex();
        aw.write_all(b"last").unwrap();
        drop(aw);
        let mut s = String::new();
        use std::io::Read as _;
        br.read_to_string(&mut s).unwrap();
        assert_eq!(s, "last");
        // Drained + closed: subsequent reads return 0.
        let mut buf = [0u8; 1];
        assert_eq!(br.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn blocking_read_wakes_on_write() {
        let (mut aw, _ar, _bw, mut br) = duplex();
        thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(50));
            let _ = aw.write_all(b"wake");
        });
        let mut buf = [0u8; 4];
        br.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"wake");
    }

    #[test]
    fn write_after_peer_drop_still_buffered() {
        // Writes into a closed direction don't error (no reader-side
        // close signal exists — mirrors a pipe with a live fd); the data
        // simply waits and is delivered when/if the peer reads.
        let (mut aw, ar, _bw, _br) = duplex();
        drop(ar);
        assert!(aw.write_all(b"ignored").is_ok());
    }
}
