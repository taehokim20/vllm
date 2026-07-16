/**
 * One-sided, CUDA-graph-safe signaling primitives for the MoE monokernel's
 * EP all-to-all *hiding* path (single-node, NVLink, EP=2).
 *
 * Design goals
 * ------------
 * 1. Fully one-sided: a producer writes payload/flag into a peer's symmetric
 *    buffer and a consumer polls that flag — no collective barrier, no host
 *    round-trip. This replaces the router all-gather / tiny all-reduce that
 *    the reference used purely for cross-rank ordering.
 *
 * 2. CUDA-graph safe: the flag value is NOT baked into the captured graph.
 *    Both the producer's stored flag and the consumer's expected flag are
 *    read from a device-resident epoch counter (``epoch_ptr``) at kernel
 *    runtime. A graph-captured op advances the epoch once per step, so each
 *    replay observes a fresh, monotonically-increasing value. A stale flag
 *    from a previous step is therefore always strictly smaller than the
 *    current epoch -> no false match, and no per-step buffer reset is needed.
 *
 * 3. Cross-process correctness (vLLM runs one process per rank): all remote
 *    stores/loads use system scope (``.sys``) so they are visible across CUDA
 *    contexts / processes over VMM/IPC-mapped memory.
 *
 * The substrate here is raw PTX over CUDA-IPC peer memory. It is kept behind
 * the Python ``ep_signal`` interface so an NVSHMEM backend can replace it if
 * multi-node/scale-out becomes a target (see ep_signal.py).
 */

#pragma once
#ifndef MOE_COMM_LL_H
#define MOE_COMM_LL_H

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace moe_monokernel {

// ============================================================================
// LL packet: 128-bit structure (4 bf16 payload + duplicated flag), matching
// the NCCL-LL format. Used by the combine peer-reduce payload path.
// ============================================================================
struct LLPacket {
  uint32_t data_low;   // lower 32 bits of payload (2 bf16 values)
  uint32_t flag1;      // flag value
  uint32_t data_high;  // upper 32 bits of payload (2 bf16 values)
  uint32_t flag2;      // flag value (duplicated for 128-bit atomicity)
};
static_assert(sizeof(LLPacket) == 16, "LLPacket must be 128 bits");

static constexpr int LL_ELEMS_PER_PACKET = 4;  // 4 bf16 per packet

// ---------------------------------------------------------------------------
// Low-level 128-bit LL store/load (constant flag). System scope for
// cross-process visibility. Kept for the combine payload path; the EP
// readiness handshake below uses the epoch-based 32-bit flag helpers instead.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void storeLL(LLPacket* dst, uint64_t val,
                                         uint32_t flag) {
  uint32_t lo = static_cast<uint32_t>(val);
  uint32_t hi = static_cast<uint32_t>(val >> 32);
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
               :
               : "l"(dst), "r"(lo), "r"(flag), "r"(hi), "r"(flag)
               : "memory");
}

__device__ __forceinline__ uint64_t readLL(const LLPacket* src,
                                            uint32_t expected_flag) {
  uint32_t d1, f1, d2, f2;
  do {
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(d1), "=r"(f1), "=r"(d2), "=r"(f2)
                 : "l"(src)
                 : "memory");
  } while (f1 != expected_flag || f2 != expected_flag);
  return static_cast<uint64_t>(d1) | (static_cast<uint64_t>(d2) << 32);
}

// ============================================================================
// CUDA-graph-safe epoch read
// ============================================================================
/**
 * @brief Read the current epoch from device memory with acquire+system scope.
 *
 * The epoch is advanced once per forward step by a graph-captured op on the
 * host side (MoELLWorkspace.advance_epoch). Reading it here (rather than
 * taking a captured constant) is what makes the flag fresh on every replay.
 */
__device__ __forceinline__ uint32_t load_epoch(const uint32_t* epoch_ptr) {
  uint32_t e;
  asm volatile("ld.acquire.sys.global.b32 %0, [%1];\n"
               : "=r"(e)
               : "l"(epoch_ptr)
               : "memory");
  return e;
}

// ============================================================================
// EP readiness handshake — one-sided, graph-safe (32-bit flag == epoch)
// ============================================================================
/**
 * @brief Publish "my staged tokens are ready" to my own flag slot.
 *
 * Call AFTER this rank has written its owned rows into the dispatch (or
 * combine) staging region. A system-scope release fence orders the staged
 * data before the flag, so a peer that observes the flag is guaranteed to see
 * the data. The stored value is the current device epoch.
 *
 * @param my_flag_slot  Pointer to THIS rank's flag slot (in its own buffer).
 * @param epoch_ptr     Device epoch counter (flag base for this step).
 */
__device__ __forceinline__ void ep_set_ready(uint32_t* my_flag_slot,
                                              const uint32_t* epoch_ptr) {
  uint32_t epoch = load_epoch(epoch_ptr);
  // Ensure the staged activation writes are globally visible before the flag.
  __threadfence_system();
  asm volatile("st.release.sys.global.b32 [%0], %1;\n"
               :
               : "l"(my_flag_slot), "r"(epoch)
               : "memory");
}

/**
 * @brief Spin until a peer has published readiness for this step.
 *
 * Polls the peer's (peer-mapped) flag slot until it is >= the current epoch.
 * ``>=`` (not ``==``) tolerates a peer that has already advanced to a later
 * step; because the epoch is monotonic, a stale value from a previous step is
 * strictly smaller and never matches. Self-ordering: the acquire load pairs
 * with the producer's release store to establish happens-before across
 * processes, so the peer's staged data is visible once this returns.
 *
 * @param peer_flag_slot  Peer-mapped pointer to the PEER's flag slot.
 * @param epoch_ptr       Device epoch counter (expected flag for this step).
 */
__device__ __forceinline__ void ep_wait_ready(const uint32_t* peer_flag_slot,
                                               const uint32_t* epoch_ptr) {
  uint32_t expected = load_epoch(epoch_ptr);
  uint32_t seen;
  do {
    asm volatile("ld.acquire.sys.global.b32 %0, [%1];\n"
                 : "=r"(seen)
                 : "l"(peer_flag_slot)
                 : "memory");
  } while (seen < expected);
}

// ---------------------------------------------------------------------------
// Warp helper: only lane 0 of a chosen warp should drive the handshake to
// avoid every thread hammering the flag. Callers gate on (threadIdx.x == 0).
// ---------------------------------------------------------------------------

}  // namespace moe_monokernel

#endif  // MOE_COMM_LL_H
