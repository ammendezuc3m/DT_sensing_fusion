#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import uhd
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
REPO_DIR = Path.cwd() / "src/python/ssb_python"
if REPO_DIR.exists() and str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from detect_pss_offline import build_pss_reference_grid, correlate_reference, ofdm_modulate_grid
from profile_online_datassb_pipeline import capture_one_block, make_rx_streamer

MODES = {
    "quick": dict(coarse_ms=20.0, confirm_ms=40.0, confirm_attempts=2,
                  min_hits=2, max_candidates=8, excess_db=5.0),
    "balanced": dict(coarse_ms=40.0, confirm_ms=80.0, confirm_attempts=3,
                     min_hits=2, max_candidates=14, excess_db=3.0),
    "exhaustive": dict(coarse_ms=160.0, confirm_ms=160.0, confirm_attempts=4,
                       min_hits=2, max_candidates=24, excess_db=1.5),
}

@dataclass(frozen=True)
class RasterPoint:
    gscn: int
    frequency_hz: float

@dataclass
class Detection:
    frequency_mhz: float
    gscn: int
    scs_khz: int
    nid2: int
    coarse_metric: float
    coarse_center_mhz: float
    coarse_power_db: float
    coarse_excess_db: float
    confirm_hits: int = 0
    confirm_attempts: int = 0
    confirm_metric_median: float = float("nan")
    confirm_metric_max: float = float("nan")
    confirmed: bool = False


def args_parser():
    p = argparse.ArgumentParser(description="Fast unknown-band 5G NR FR1 SSB discovery")
    p.add_argument("--serial", default="")
    p.add_argument("--rx-channel", type=int, default=0)
    p.add_argument("--antenna", default="")
    p.add_argument("--gain", type=float, default=60.0)
    p.add_argument("--rate", type=float, default=30.72e6)
    p.add_argument("--start-mhz", type=float, default=70.0)
    p.add_argument("--stop-mhz", type=float, default=6000.0)
    p.add_argument("--mode", choices=MODES, default="quick")
    p.add_argument("--settle-sec", type=float, default=0.035)
    p.add_argument("--usable-fraction", type=float, default=0.78)
    p.add_argument("--overlap", type=float, default=0.12)
    p.add_argument("--scs", choices=["auto", "15", "30"], default="auto")
    p.add_argument("--force-nid2", type=int, choices=[0,1,2], default=None)
    p.add_argument("--min-pss-metric", type=float, default=0.35)
    p.add_argument("--confirm-min-pss-metric", type=float, default=0.45)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--out-dir", default="results/ssb_auto_scan")
    p.add_argument("--prefix", default="ssb_auto_scan")
    p.add_argument("--test-all-above-3ghz", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    if not (0.2 <= a.usable_fraction <= 0.95): p.error("bad --usable-fraction")
    if not (0 <= a.overlap < 0.8): p.error("bad --overlap")
    if a.stop_mhz <= a.start_mhz: p.error("stop must exceed start")
    if a.rate < 15.36e6: p.error("rate must be >= 15.36e6")
    return a


def global_raster(start_hz, stop_hz):
    pts = []
    if start_hz < 3e9:
        lo, hi = max(start_hz, 0.0), min(stop_hz, 3e9)
        n0 = max(1, math.floor(lo/1.2e6)-1)
        n1 = min(2499, math.ceil(hi/1.2e6)+1)
        for n in range(n0, n1+1):
            for m in (1,3,5):
                f = n*1.2e6 + m*50e3
                if lo <= f < hi:
                    pts.append(RasterPoint(int(3*n + (m-3)/2), f))
    if stop_hz >= 3e9:
        lo, hi = max(start_hz, 3e9), min(stop_hz, 24.25e9)
        n0 = max(0, math.ceil((lo-3e9)/1.44e6 - 1e-12))
        n1 = min(14756, math.floor((hi-3e9)/1.44e6 + 1e-12))
        for n in range(n0, n1+1):
            pts.append(RasterPoint(7499+n, 3e9+n*1.44e6))
    return sorted(pts, key=lambda x: x.frequency_hz)


def tune_centers(start_hz, stop_hz, rate, usable_fraction, overlap):
    usable = rate*usable_fraction
    step = usable*(1-overlap)
    if stop_hz-start_hz <= usable:
        return [(start_hz+stop_hz)/2], usable
    centers, c, last = [], start_hz+usable/2, stop_hz-usable/2
    while c < last:
        centers.append(c); c += step
    centers.append(last)
    return centers, usable


def configure(a):
    dev = f"serial={a.serial}" if a.serial else ""
    u = uhd.usrp.MultiUSRP(dev)
    ch = a.rx_channel
    u.set_rx_rate(a.rate, ch); u.set_rx_gain(a.gain, ch)
    if a.antenna: u.set_rx_antenna(a.antenna, ch)
    time.sleep(a.settle_sec)
    return u, float(u.get_rx_rate(ch))


def psd(w, rate, nfft=4096):
    nfft = min(nfft, 2**int(math.floor(math.log2(len(w)))))
    blocks = len(w)//nfft
    z = w[:blocks*nfft].reshape(blocks,nfft)*np.hanning(nfft)[None,:]
    p = np.mean(np.abs(np.fft.fftshift(np.fft.fft(z,axis=1),axes=1))**2,axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(nfft,1/rate))
    return f,p


def band_power_db(f,p,offset,bw=4.2e6):
    m = np.abs(f-offset) <= bw/2
    return 10*np.log10(max(float(np.mean(p[m])) if np.any(m) else 1e-30,1e-30))


def selected_points(raster, center, usable, f, p, mode, test_all_above):
    half=usable/2
    pts=[x for x in raster if center-half <= x.frequency_hz <= center+half]
    scored=[(x,band_power_db(f,p,x.frequency_hz-center)) for x in pts]
    if not scored: return []
    med=float(np.median([v for _,v in scored]))
    ranked=sorted([(x,v,v-med) for x,v in scored],key=lambda q:q[1],reverse=True)
    if test_all_above and center >= 3e9: return ranked
    out=[x for x in ranked if x[2] >= mode["excess_db"]]
    if len(out)<min(3,len(ranked)): out=ranked[:min(3,len(ranked))]
    return out[:mode["max_candidates"]]


def shift_resample(w, offset, in_rate):
    n=np.arange(len(w),dtype=np.float64)
    y=w*np.exp(-1j*2*np.pi*offset*n/in_rate).astype(np.complex64)
    ratio=in_rate/15.36e6
    q=int(round(ratio))
    if abs(ratio-q)<1e-6:
        return y if q==1 else resample_poly(y,1,q).astype(np.complex64)
    from fractions import Fraction
    r=Fraction(15.36e6/in_rate).limit_denominator(128)
    return resample_poly(y,r.numerator,r.denominator).astype(np.complex64)


def reference(nid2, scs):
    g=build_pss_reference_grid(nid2=nid2,nrb_ssb=20)
    if scs==30: return ofdm_modulate_grid(g,nfft=512,cp_lengths=[40,36])
    return ofdm_modulate_grid(g,nfft=1024,cp_lengths=[80,72])


def detect(w, scs_list, nid2_list):
    best=(-1.0,-1,-1)
    for scs in scs_list:
        for nid2 in nid2_list:
            r=reference(nid2,scs)
            if len(w)<=len(r): continue
            m=correlate_reference(w,r)
            val=float(np.max(m))
            if val>best[0]: best=(val,scs,nid2)
    return best


def scs_list(a): return [15,30] if a.scs=="auto" else [int(a.scs)]
def nid2_list(a): return [0,1,2] if a.force_nid2 is None else [a.force_nid2]


def dedup(ds, tol_khz=120):
    out=[]
    for d in sorted(ds,key=lambda x:x.coarse_metric,reverse=True):
        if not any(abs(d.frequency_mhz-k.frequency_mhz)*1000<=tol_khz and d.scs_khz==k.scs_khz and d.nid2==k.nid2 for k in out):
            out.append(d)
    return out


def main():
    a=args_parser(); mode=MODES[a.mode]
    start,stop=a.start_mhz*1e6,a.stop_mhz*1e6
    raster=global_raster(start,stop)
    u,rate=configure(a)
    centers,usable=tune_centers(start,stop,rate,a.usable_fraction,a.overlap)
    rx=make_rx_streamer(u,a.rx_channel); maxs=rx.get_max_num_samps()
    coarse_n=int(rate*mode["coarse_ms"]*1e-3)
    prelim=[]; tune_rows=[]; t0=time.perf_counter()
    print("=== AUTO 5G NR SSB SCAN ===")
    print(f"range={a.start_mhz:.1f}-{a.stop_mhz:.1f} MHz mode={a.mode} tunes={len(centers)} raster={len(raster)}")
    print(f"rate={rate/1e6:.2f} Msps usable={usable/1e6:.2f} MHz SCS={scs_list(a)} NID2={nid2_list(a)}")
    for i,req in enumerate(centers,1):
        u.set_rx_freq(uhd.types.TuneRequest(req),a.rx_channel); time.sleep(a.settle_sec)
        center=float(u.get_rx_freq(a.rx_channel)); row={"index":i,"center_mhz":center/1e6,"tested":0,"hits":0,"error":""}
        try:
            w=capture_one_block(rx,coarse_n,maxs); ff,pp=psd(w,rate)
            cand=selected_points(raster,center,usable,ff,pp,mode,a.test_all_above_3ghz); row["tested"]=len(cand)
            for point,pdb,exdb in cand:
                y=shift_resample(w,point.frequency_hz-center,rate)
                metric,scs,nid2=detect(y,scs_list(a),nid2_list(a))
                if metric>=a.min_pss_metric:
                    d=Detection(point.frequency_hz/1e6,point.gscn,scs,nid2,metric,center/1e6,pdb,exdb)
                    prelim.append(d); row["hits"]+=1
                    print(f"  HIT {d.frequency_mhz:.6f} MHz GSCN={d.gscn} SCS={scs} NID2={nid2} metric={metric:.3f}")
        except Exception as e: row["error"]=str(e)
        tune_rows.append(row)
        if i==1 or i==len(centers) or i%a.progress_every==0:
            print(f"[{i:04d}/{len(centers):04d}] {center/1e6:10.3f} MHz tested={row['tested']:3d} hits={row['hits']:2d} {row['error']}")

    prelim=dedup(prelim)
    print(f"\n=== CONFIRMATION ({len(prelim)} candidates) ===")
    confirmed=[]; confirm_n=int(rate*mode["confirm_ms"]*1e-3)
    for i,d in enumerate(prelim,1):
        target=d.frequency_mhz*1e6
        u.set_rx_freq(uhd.types.TuneRequest(target),a.rx_channel); time.sleep(max(a.settle_sec,0.05))
        center=float(u.get_rx_freq(a.rx_channel)); vals=[]; hits=0
        for _ in range(mode["confirm_attempts"]):
            w=capture_one_block(rx,confirm_n,maxs)
            y=shift_resample(w,target-center,rate)
            metric,_,_=detect(y,[d.scs_khz],[d.nid2]); vals.append(metric)
            hits += metric>=a.confirm_min_pss_metric
        d.confirm_hits=int(hits); d.confirm_attempts=mode["confirm_attempts"]
        d.confirm_metric_median=float(np.median(vals)); d.confirm_metric_max=float(np.max(vals))
        d.confirmed=hits>=mode["min_hits"]
        print(f"[{i:02d}/{len(prelim):02d}] {d.frequency_mhz:.6f} MHz SCS={d.scs_khz} NID2={d.nid2} hits={hits}/{mode['confirm_attempts']} median={d.confirm_metric_median:.3f} {'CONFIRMED' if d.confirmed else 'REJECTED'}")
        if d.confirmed: confirmed.append(d)

    confirmed.sort(key=lambda d:(d.confirm_hits,d.confirm_metric_median),reverse=True)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); stamp=time.strftime("%Y%m%d_%H%M%S")
    j=out/f"{a.prefix}_{stamp}.json"; c=out/f"{a.prefix}_{stamp}.csv"
    payload={"configuration":vars(a),"actual_rate_hz":rate,"usable_bw_hz":usable,"num_tunes":len(centers),"elapsed_s":time.perf_counter()-t0,"confirmed":[asdict(d) for d in confirmed],"preliminary":[asdict(d) for d in prelim],"tunes":tune_rows}
    j.write_text(json.dumps(payload,indent=2,allow_nan=True),encoding="utf-8")
    fields=list(asdict(Detection(0,0,0,0,0,0,0,0)).keys())
    with c.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([asdict(d) for d in confirmed])
    print("\n=== CONFIRMED SSBS ===")
    if not confirmed: print("None")
    for rank,d in enumerate(confirmed,1):
        print(f"{rank:2d}. SSREF={d.frequency_mhz:.6f} MHz GSCN={d.gscn} SCS={d.scs_khz} kHz NID2={d.nid2} median={d.confirm_metric_median:.3f}")
    print(f"elapsed={payload['elapsed_s']:.1f}s\nJSON={j}\nCSV={c}")

if __name__=="__main__": main()
