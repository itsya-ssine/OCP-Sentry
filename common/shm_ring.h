/* shm_ring.h — a lock-free "latest frame" slot in POSIX shared memory.
 *
 * This is an OPTIONAL, opt-in low-latency alternative to routing raw
 * frames through Redis between services/ingest and services/inference
 * when both run on the same host (the default docker-compose.yml setup).
 * Redis stays the transport for detections/events either way — those are
 * low-volume and benefit from Redis's persistence/consumer-group
 * features; this is specifically for the high-volume, latency-sensitive
 * raw-frame handoff.
 *
 * Design: one fixed-size slot per camera, holding only the single most
 * recent frame — not a queue. inference_worker.py always wants "the
 * newest frame available right now", never a backlog, so there's no
 * benefit to buffering older frames, and it lets this skip locks/
 * semaphores entirely in favor of a seqlock:
 *
 *   - the writer increments `seq` to an odd value before writing (readers
 *     use this to detect "write in progress, don't trust this data")
 *   - the writer writes ts/width/height/jpg_len/jpg
 *   - the writer increments `seq` again to an even value ("stable")
 *   - a reader reads seq, copies the fields, re-reads seq, and retries if
 *     seq changed (a "torn" read straddling a write) or was odd
 *
 * This makes writes wait-free and reads either succeed cleanly or retry a
 * few times — no blocking, no priority inversion, no possibility of a
 * crashed reader/writer holding a lock forever. It supports one writer
 * and any number of readers on the same slot.
 *
 * Frames are capped at SHM_RING_MAX_FRAME bytes; a larger frame is
 * truncated by the writer rather than overflowing the segment (see
 * shm_ring_write in shm_ring.c) — sanity-check this against your actual
 * detect_url resolution / JPEG quality if you raise frame_sample_fps or
 * capture resolution significantly.
 */
#ifndef SHM_RING_H
#define SHM_RING_H

#include <stddef.h>
#include <stdint.h>

#define SHM_RING_MAGIC ((uint32_t)0x53484d31u) /* "SHM1" */
#define SHM_RING_MAX_FRAME (512u * 1024u)

typedef struct {
    uint32_t magic;
    volatile uint32_t seq; /* seqlock: even = stable, odd = writer in progress */
    uint64_t frame_id;     /* monotonically increasing; readers compare against last-seen */
    double ts;              /* unix timestamp (seconds) the frame was captured */
    uint32_t width;
    uint32_t height;
    uint32_t jpg_len;       /* valid bytes in jpg[] */
    unsigned char jpg[SHM_RING_MAX_FRAME];
} shm_ring_slot_t;

/* Creates (or re-creates) the named shared memory segment and maps it.
 * `name` must start with '/' per POSIX shm_open rules, e.g. "/nvr_frame_webcam".
 * Returns 0 on success, -1 on failure (check errno). */
int shm_ring_create(const char *name, shm_ring_slot_t **out_slot);

/* Opens an already-created segment (fails if the writer hasn't called
 * shm_ring_create yet — the reader side should retry/backoff on failure
 * rather than treat it as fatal, since ingest and inference start up
 * independently). Returns 0 on success, -1 on failure. */
int shm_ring_open(const char *name, shm_ring_slot_t **out_slot);

/* Unmaps the slot. Does not remove the underlying shared memory object —
 * call shm_ring_unlink for that (the writer/owner's responsibility, not
 * every reader's). */
void shm_ring_close(shm_ring_slot_t *slot);

/* Removes the named shared memory object. Only the writer (owner) should
 * call this, on clean shutdown — readers just shm_ring_close(). */
void shm_ring_unlink(const char *name);

/* Overwrites the slot with a new frame. Truncates jpg_len to
 * SHM_RING_MAX_FRAME if the caller passes something larger. Wait-free:
 * never blocks on readers. */
void shm_ring_write(shm_ring_slot_t *slot, double ts, uint32_t width, uint32_t height,
                     const unsigned char *jpg, uint32_t jpg_len);

/* If the slot holds a frame newer than `last_seen_id`, copies it into the
 * caller-supplied buffer (up to out_jpg_cap bytes) and returns 1. Returns
 * 0 if nothing newer is available yet, or if it couldn't get a clean
 * (non-torn) read after a few retries — callers should just poll again
 * shortly rather than treat 0 as an error. */
int shm_ring_read_if_newer(shm_ring_slot_t *slot, uint64_t last_seen_id,
                            uint64_t *out_frame_id, double *out_ts,
                            uint32_t *out_width, uint32_t *out_height,
                            unsigned char *out_jpg, uint32_t *out_jpg_len, uint32_t out_jpg_cap);

#endif /* SHM_RING_H */
