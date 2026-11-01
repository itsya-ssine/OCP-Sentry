/* mjpeg_split.c
 *
 * Sits between ffmpeg and services/ingest/ingest.py in the detect-stream
 * pipeline. ffmpeg's `-f mjpeg pipe:1` output is just concatenated JPEGs
 * with no framing of its own, so something has to scan for SOI (0xFFD8) /
 * EOI (0xFFD9) markers to split it into individual frames. The original
 * implementation did this in pure Python, re-slicing a `bytes` buffer
 * (i.e. reallocating + copying the remainder) on every single frame,
 * continuously, per camera, for the life of the process. This C version
 * does the same scan with a fixed buffer and memmove (no allocation churn)
 * and writes each frame to stdout as a simple [4-byte LE length][JPEG
 * bytes] record, which ingest.py reads with two fixed-size reads instead
 * of doing any scanning itself.
 *
 * Usage: ffmpeg ... -f mjpeg pipe:1 | mjpeg_split | (consumed by Python)
 * ingest.py wires this up directly via subprocess, not a shell pipe (see
 * _detect_loop in ingest.py).
 *
 * Exits (and lets ingest.py's restart-with-backoff loop respawn everything)
 * on EOF from stdin, a write error on stdout, or a JPEG stream that never
 * produces a complete frame within BUF_CAPACITY bytes (treated as a
 * corrupt/desynced stream rather than grown without bound).
 */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Generous for a single sampled JPEG frame (this pipeline samples at a few
 * fps, not full framerate, and frames are already fairly low quality
 * -q:v 5). If your detect_url is unusually high resolution and a frame
 * genuinely exceeds this, bump it here and rebuild the ingest image. */
#define BUF_CAPACITY (2 * 1024 * 1024)

static unsigned char buf[BUF_CAPACITY];
static size_t buf_len = 0; /* valid bytes currently in buf[0..buf_len) */

static const unsigned char SOI[2] = {0xFF, 0xD8};
static const unsigned char EOI[2] = {0xFF, 0xD9};

/* Index of the first occurrence of `needle` (2 bytes) in buf[from..buf_len),
 * or -1 if not found. */
static long find_marker(size_t from, const unsigned char *needle) {
    if (from + 2 > buf_len) return -1;
    for (size_t i = from; i + 1 < buf_len; i++) {
        if (buf[i] == needle[0] && buf[i + 1] == needle[1]) return (long)i;
    }
    return -1;
}

static int write_all(int fd, const void *data, size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

int main(void) {
    for (;;) {
        if (buf_len >= BUF_CAPACITY) {
            fprintf(stderr, "[mjpeg_split] no complete frame within %d bytes, resyncing\n", BUF_CAPACITY);
            buf_len = 0;
        }

        ssize_t n;
        do {
            n = read(STDIN_FILENO, buf + buf_len, BUF_CAPACITY - buf_len);
        } while (n < 0 && errno == EINTR);
        if (n <= 0) break; /* EOF or read error: ffmpeg exited, let the caller restart us */
        buf_len += (size_t)n;

        size_t scan_pos = 0;
        for (;;) {
            long start = find_marker(scan_pos, SOI);
            if (start < 0) {
                /* No frame start anywhere in the unscanned tail: none of it
                 * can become part of a future frame, so drop it all. */
                buf_len = 0;
                break;
            }
            long end = find_marker((size_t)start + 2, EOI);
            if (end < 0) {
                /* Frame start found but not yet complete: keep from `start`
                 * onward and wait for more data on the next read(). */
                if (start > 0) {
                    memmove(buf, buf + start, buf_len - (size_t)start);
                    buf_len -= (size_t)start;
                }
                break;
            }

            size_t frame_len = (size_t)end + 2 - (size_t)start;
            uint32_t len_le = (uint32_t)frame_len; /* little-endian on every platform this project targets (x86_64/arm64) */

            if (write_all(STDOUT_FILENO, &len_le, sizeof(len_le)) != 0) return 1;
            if (write_all(STDOUT_FILENO, buf + start, frame_len) != 0) return 1;

            scan_pos = (size_t)end + 2;
        }
    }
    return 0;
}
