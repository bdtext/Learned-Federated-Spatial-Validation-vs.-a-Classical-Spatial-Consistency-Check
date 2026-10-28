"""Verification of federated aggregation by independent re-computation (software simulation, no ledger).
What it tests: (1) canonical serialization (little-endian float32, fixed reduction order) makes K independent verifier
processes produce bit-identical SHA-256 digests of the aggregate for mean, coordinate median, multi-Krum and FLTrust;
(2) a corrupted coordinator proposal is rejected by the attestation quorum q = f+1 and attributed; (3) a withheld
or corrupted input artifact is detected by its digest; (4) verification compute per organization vs model size.
This simulates the protocol logic only; it is NOT a Hyperledger Fabric measurement.
"""
import os, sys, json, time, hashlib
import numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from qc import fl as FL
from run_detection import jdump

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "govsim")

def canonical(vec):
    return np.ascontiguousarray(vec.detach().cpu().numpy().astype("<f4")).tobytes()

def digest(vec):
    return hashlib.sha256(canonical(vec)).hexdigest()

def aggregate(rule, U, f=1, u0=None):
    if rule == "mean":
        return U.mean(0)
    if rule == "median":
        return FL.agg_median(U)
    if rule == "multikrum":
        return FL.agg_multikrum(U, f)[0]
    if rule == "fltrust":
        return FL.agg_fltrust(U, u0)[0]

def verifier(args):
    """One organization re-computes the aggregate from the committed artifacts (bytes) and returns the digest and time."""
    rule, blobs, f, u0_blob, seed = args
    torch.manual_seed(seed)  # different process state on purpose
    U = torch.stack([torch.from_numpy(np.frombuffer(b, dtype="<f4").copy()) for b in blobs])
    u0 = torch.from_numpy(np.frombuffer(u0_blob, dtype="<f4").copy()) if u0_blob is not None else None
    t0 = time.perf_counter()
    agg = aggregate(rule, U, f, u0)
    d = digest(agg)
    return d, time.perf_counter() - t0

def main():
    os.makedirs(RES, exist_ok=True)
    out = {}
    rng = np.random.default_rng(0)
    for n_params in (23494, 100000, 1500000):
        for K in (3, 7):
            f = (K - 1) // 2 if K > 3 else 1
            U = torch.from_numpy(rng.normal(0, 0.01, size=(K, n_params)).astype(np.float32))
            u0 = torch.from_numpy(rng.normal(0, 0.01, size=n_params).astype(np.float32))
            blobs = [canonical(u) for u in U]
            for rule in ("mean", "median", "multikrum", "fltrust"):
                if rule == "multikrum" and K < 2 * f + 3:
                    fk = max((K - 3) // 2, 0)
                    if fk < 1: continue
                else:
                    fk = f
                with Pool(K) as pool:
                    results = pool.map(verifier, [(rule, blobs, fk, canonical(u0) if rule == "fltrust" else None, 100 + k) for k in range(K)])
                digests = [r[0] for r in results]; times = [r[1] for r in results]
                identical = len(set(digests)) == 1
                # tamper test: coordinator proposes a digest of a biased aggregate
                biased = aggregate(rule, U, fk, u0) + 1e-3
                prop = digest(biased)
                n_match = sum(d == prop for d in digests); q = fk + 1
                verified = n_match >= q and identical
                honest_digest = max(set(digests), key=digests.count)
                failed_attributed = (not verified) and digests.count(honest_digest) >= q
                # withheld/corrupted artifact: flip one byte of one committed blob -> digest mismatch at fetch
                corrupt = bytearray(blobs[0]); corrupt[0] ^= 0xFF
                mismatch = hashlib.sha256(bytes(corrupt)).hexdigest() != hashlib.sha256(blobs[0]).hexdigest()
                out[f"p{n_params}_K{K}_{rule}"] = dict(n_params=n_params, K=K, f=fk, rule=rule, verifiers_identical=identical,
                                                     verify_ms_median=float(np.median(times) * 1000), verify_ms_max=float(np.max(times) * 1000),
                                                     tampered_proposal_rejected=not verified, coordinator_attributed=failed_attributed,
                                                     corrupted_artifact_detected=mismatch, payload_bytes_digest=64)
                print(out[f"p{n_params}_K{K}_{rule}"], flush=True)
    jdump(out, os.path.join(RES, "govsim.json"))

if __name__ == "__main__":
    main()
