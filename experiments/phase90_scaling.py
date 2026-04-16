"""
Phase 90: L-NCA Capacity Scaling Law

Scale hidden_ch: 16 -> 32 -> 64 -> 128
Find the sweet spot: max Exact Match within 500ms budget.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

N_COLORS = 10
NCA_STEPS = 10
GS = 8
EPOCHS = 80
LR = 1e-3
BS = 32
N_PER_TASK = 1000
N_TEST = 200
HIDDEN_SIZES = [16, 32, 64, 128]


def to_onehot(g, nc=N_COLORS):
    h, w = g.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): o[c] = (g == c).astype(np.float32)
    return o

def gen_gravity(n, gs=8, seed=None):
    rng = np.random.RandomState(seed); ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2,5)): r,c=rng.randint(0,gs,size=2); g[r,c]=1
        for _ in range(rng.randint(1,4)):
            r,c=rng.randint(0,gs,size=2)
            if g[r,c]==0: g[r,c]=2
        res = np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]==2: res[r,c]=2
        for c in range(gs):
            cnt=sum(1 for r in range(gs) if g[r,c]==1); row,pl=gs-1,0
            while pl<cnt and row>=0:
                if res[row,c]==0: res[row,c]=1; pl+=1
                row-=1
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)), torch.tensor(np.array(tgs))

def gen_expand(n, gs=8, seed=None):
    rng = np.random.RandomState(seed); ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2,5)):
            y,x=rng.randint(1,gs-1,size=2); c=rng.randint(1,5)
            if g[y,x]==0: g[y,x]=c
        res = g.copy()
        for y in range(gs):
            for x in range(gs):
                if g[y,x]>0:
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny,nx=y+dy,x+dx
                        if 0<=ny<gs and 0<=nx<gs and res[ny,nx]==0: res[ny,nx]=g[y,x]
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)), torch.tensor(np.array(tgs))

def gen_color_invert(n, gs=8, seed=None):
    rng = np.random.RandomState(seed); ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(3,8)): r,c=rng.randint(0,gs,size=2); g[r,c]=rng.randint(1,4)
        res = g.copy(); res[g==1]=2; res[g==2]=1
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)), torch.tensor(np.array(tgs))

def gen_move_right(n, gs=8, seed=None):
    rng = np.random.RandomState(seed); ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2,6)): r,c=rng.randint(0,gs,size=2); g[r,c]=rng.randint(1,5)
        res = np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0: res[r,(c+1)%gs]=g[r,c]
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)), torch.tensor(np.array(tgs))

def gen_fill_border(n, gs=8, seed=None):
    rng = np.random.RandomState(seed); ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2,6)): r,c=rng.randint(0,gs,size=2); g[r,c]=rng.randint(1,5)
        res = g.copy(); res[0,:]=3; res[-1,:]=3; res[:,0]=3; res[:,-1]=3
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)), torch.tensor(np.array(tgs))

ALL_TASKS = {'gravity':gen_gravity,'expand':gen_expand,'color_invert':gen_color_invert,
             'move_right':gen_move_right,'fill_border':gen_fill_border}


class LNCA(nn.Module):
    def __init__(self, nc=10, hc=16):
        super().__init__()
        self.hc = hc
        self.perceive = nn.Conv2d(nc+hc, hc*2, 3, padding=1)
        self.update = nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate = nn.Conv2d(nc+hc*2, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout = nn.Conv2d(hc, nc, 1)
    def forward(self, x, n_steps=NCA_STEPS):
        b,c,h,w = x.shape; state = torch.zeros(b,self.hc,h,w,device=x.device)
        for _ in range(n_steps):
            combined = torch.cat([x,state],1); delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x,state,delta],1)
            beta = torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state = beta*state + (1-beta)*delta
        return self.readout(state)


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase90_scaling.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment':'Phase 90: Capacity Scaling','timestamp':datetime.now().isoformat(),
                   'results':results}, f, indent=2, default=str)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("="*70); print("Phase 90: L-NCA Capacity Scaling Law")
    print(f"  hidden_ch: {HIDDEN_SIZES}"); print("="*70)

    # Generate all tasks
    all_x, all_y = [], []
    test_data = {}
    for tn, fn in ALL_TASKS.items():
        x, y = fn(N_PER_TASK, GS, seed=SEED); all_x.append(x); all_y.append(y)
        tx, ty = fn(N_TEST, GS, seed=SEED+1); test_data[tn] = (tx, ty)
    x_train, y_train = torch.cat(all_x), torch.cat(all_y)
    n = x_train.size(0)

    results = {}
    for hc in HIDDEN_SIZES:
        print(f"\n  === hidden_ch={hc} ===")
        model = LNCA(hc=hc).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            model.train(); perm = torch.randperm(n)
            for i in range(0, n, BS):
                idx = perm[i:i+BS]
                xb, yb = x_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
                opt.zero_grad(); F.cross_entropy(model(xb), yb).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if (epoch+1)%20==0:
                model.eval()
                with torch.no_grad():
                    p=model(x_train[:200].to(DEVICE)).argmax(1)
                    acc=(p==y_train[:200].to(DEVICE)).float().mean()
                print(f"    Epoch {epoch+1}: pixel={acc*100:.1f}%")

        # Evaluate per-task
        task_results = {}
        for tn, (tx, ty) in test_data.items():
            model.eval()
            with torch.no_grad():
                preds = model(tx.to(DEVICE)).argmax(1); target = ty.to(DEVICE)
                pixel = (preds==target).float().mean().item()
                exact = (preds.reshape(N_TEST,-1)==target.reshape(N_TEST,-1)).all(1).float().mean().item()
            task_results[tn] = {'pixel':pixel,'exact':exact}
            print(f"    {tn:15s}: pixel={pixel*100:.1f}% exact={exact*100:.1f}%")

        # Latency benchmark
        model.eval(); x1 = test_data['gravity'][0][:1].to(DEVICE)
        for _ in range(5): model(x1)
        times = []
        for _ in range(50):
            t0=time.perf_counter()
            with torch.no_grad(): model(x1)
            times.append((time.perf_counter()-t0)*1000)
        lat = np.mean(times)
        print(f"    Latency: {lat:.1f}ms")

        avg_exact = np.mean([v['exact'] for v in task_results.values()])
        results[f"hc{hc}"] = {'hidden_ch':hc,'n_params':n_params,'latency_ms':lat,
                              'avg_exact':avg_exact,'tasks':task_results}
        _save(results)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n{'='*70}"); print("GRAND SUMMARY: Scaling Law"); print(f"{'='*70}")
    for k, r in results.items():
        print(f"  {k}: params={r['n_params']:,} avg_exact={r['avg_exact']*100:.1f}% lat={r['latency_ms']:.1f}ms")

    _gen_fig(results); print("\nPhase 90 complete!")

def _gen_fig(results):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        entries = sorted(results.values(), key=lambda e: e['hidden_ch'])
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
        hcs = [e['hidden_ch'] for e in entries]
        exacts = [e['avg_exact']*100 for e in entries]
        lats = [e['latency_ms'] for e in entries]
        params = [e['n_params'] for e in entries]
        ax1.plot(hcs, exacts, 'o-', color='#EC4899', linewidth=2, markersize=10)
        for h,e,p in zip(hcs,exacts,params): ax1.annotate(f'{p:,}p\n{e:.0f}%',(h,e),textcoords="offset points",xytext=(0,10),ha='center',fontsize=8)
        ax1.set_xlabel('hidden_ch'); ax1.set_ylabel('Avg Exact Match (%)'); ax1.set_title('Capacity vs Accuracy',fontweight='bold'); ax1.grid(alpha=0.3)
        ax2.plot(hcs, lats, 's-', color='#3B82F6', linewidth=2, markersize=10)
        ax2.axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Budget')
        ax2.set_xlabel('hidden_ch'); ax2.set_ylabel('Latency (ms)'); ax2.set_title('Capacity vs Speed',fontweight='bold'); ax2.legend(); ax2.grid(alpha=0.3)
        fig.suptitle('Phase 90: L-NCA Capacity Scaling Law',fontsize=13,fontweight='bold')
        plt.tight_layout(); os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR,"phase90_scaling.png"),bbox_inches='tight',dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e: print(f"  Figure error: {e}")

if __name__ == '__main__': main()
