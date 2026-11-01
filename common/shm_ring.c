/* shm_ring.c — see shm_ring.h for the design rationale (seqlock, single
 * most-recent-frame slot, one writer / many readers). */
#include "shm_ring.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static size_t slot_size(void) {
    return sizeof(shm_ring_slot_t);
}

int shm_ring_create(const char *name, shm_ring_slot_t **out_slot) {
    int fd = shm_open(name, O_CREAT | O_RDWR, 0660);
    if (fd < 0) return -1;

    if (ftruncate(fd, (off_t)slot_size()) != 0) {
        close(fd);
        return -1;
    }

    void *addr = mmap(NULL, slot_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd); /* the mapping stays valid after closing the fd */
    if (addr == MAP_FAILED) return -1;

    shm_ring_slot_t *slot = (shm_ring_slot_t *)addr;
    memset(slot, 0, sizeof(*slot));
    slot->magic = SHM_RING_MAGIC;
    slot->seq = 0;

    *out_slot = slot;
    return 0;
}

int shm_ring_open(const char *name, shm_ring_slot_t **out_slot) {
    int fd = shm_open(name, O_RDWR, 0660);
    if (fd < 0) return -1;

    void *addr = mmap(NULL, slot_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return -1;

    *out_slot = (shm_ring_slot_t *)addr;
    return 0;
}

void shm_ring_close(shm_ring_slot_t *slot) {
    if (slot != NULL) munmap(slot, slot_size());
}

void shm_ring_unlink(const char *name) {
    shm_unlink(name);
}

void shm_ring_write(shm_ring_slot_t *slot, double ts, uint32_t width, uint32_t height,
                     const unsigned char *jpg, uint32_t jpg_len) {
    if (jpg_len > SHM_RING_MAX_FRAME) jpg_len = SHM_RING_MAX_FRAME;

    uint32_t seq = slot->seq;
    slot->seq = seq + 1; /* odd: readers should back off */
    __sync_synchronize();

    slot->frame_id += 1;
    slot->ts = ts;
    slot->width = width;
    slot->height = height;
    slot->jpg_len = jpg_len;
    memcpy(slot->jpg, jpg, jpg_len);

    __sync_synchronize();
    slot->seq = seq + 2; /* even: stable again */
}

int shm_ring_read_if_newer(shm_ring_slot_t *slot, uint64_t last_seen_id,
                            uint64_t *out_frame_id, double *out_ts,
                            uint32_t *out_width, uint32_t *out_height,
                            unsigned char *out_jpg, uint32_t *out_jpg_len, uint32_t out_jpg_cap) {
    for (int attempt = 0; attempt < 8; attempt++) {
        uint32_t seq1 = slot->seq;
        if (seq1 & 1u) continue; /* writer mid-update, retry */

        uint64_t frame_id = slot->frame_id;
        if (frame_id == last_seen_id) return 0; /* nothing new since last read */

        double ts = slot->ts;
        uint32_t width = slot->width;
        uint32_t height = slot->height;
        uint32_t jpg_len = slot->jpg_len;
        uint32_t copy_len = jpg_len < out_jpg_cap ? jpg_len : out_jpg_cap;
        memcpy(out_jpg, slot->jpg, copy_len);

        __sync_synchronize();
        uint32_t seq2 = slot->seq;
        if (seq1 != seq2) continue; /* torn read (writer ran during our copy), retry */

        *out_frame_id = frame_id;
        *out_ts = ts;
        *out_width = width;
        *out_height = height;
        *out_jpg_len = copy_len;
        return 1;
    }
    return 0; /* gave up after retries; caller just polls again next loop iteration */
}
