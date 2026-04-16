"""
Phase 91: TTT & Inference Compute Scaling

Use the 260ms latency budget surplus:
  A) Scale TTT (Prompt Tuning) steps: 15 -> 30 -> 50
  B) Scale NCA inference steps (T): 10 -> 15 -> 20 -> 30
  C) Combined: optimal TTT + optimal T

Uses Phase 90's best model (saved as phase90_best.pt).

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy
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
N_COLORS = 10; GS = 8; BS = 32; N_TEST = 200; N_DEMO = 3
CTX_CH = 4; TTT_LR = 0.1
BEST_HC = 64  # Will use this if Phase 90 best not found
TRAIN_EPOCHS = 60; N_PER_TASK = 1000; LR = 1e-3

def to_onehot(g, nc=N_COLORS):
    h,w=g.shape; o=np.zeros((nc,h,w),dtype=np.float32)
    for c in range(nc): o[c]=(g==c).astype(np.float32)
    return o

def gen_gravity(n,gs=8,seed=None):
    rng=np.random.RandomState(seed); ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): r,c=rng.randint(0,gs,size=2); g[r,c]=1
        for _ in range(rng.randint(1,4)):
            r,c=rng.randint(0,gs,size=2)
            if g[r,c]==0: g[r,c]=2
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]==2: res[r,c]=2
        for c in range(gs):
            cnt=sum(1 for r in range(gs) if g[r,c]==1); row,pl=gs-1,0
            while pl<cnt and row>=0:
                if res[row,c]==0: res[row,c]=1; pl+=1
                row-=1
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

def gen_expand(n,gs=8,seed=None):
    rng=np.random.RandomState(seed); ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):
            y,x=rng.randint(1,gs-1,size=2); c=rng.randint(1,5)
            if g[y,x]==0: g[y,x]=c
        res=g.copy()
        for y in range(gs):
            for x in range(gs):
                if g[y,x]>0:
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny,nx=y+dy,x+dx
                        if 0<=ny<gs and 0<=nx<gs and res[ny,nx]==0: res[ny,nx]=g[y,x]
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

ALL_TASKS = {'gravity':gen_gravity, 'expand':gen_expand}


class PromptLNCA(nn.Module):
    def __init__(self, nc=10, hc=64, ctx_ch=4):
        super().__init__()
        self.hc=hc; self.ctx_proj=nn.Conv2d(ctx_ch,nc,1,bias=False)
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
        self.task_context=nn.Parameter(torch.zeros(1,ctx_ch,1,1))
    def forward(self, x, n_steps=10):
        b,c,h,w=x.shape
        ctx=self.ctx_proj(self.task_context.expand(b,-1,h,w)); x_aug=x+ctx
        state=torch.zeros(b,self.hc,h,w,device=x.device)
        for _ in range(n_steps):
            combined=torch.cat([x_aug,state],1); delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x_aug,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
        return self.readout(state)
    def freeze_backbone(self):
        for name,p in self.named_parameters():
            if name!='task_context': p.requires_grad=False


def train_foundation(hc):
    """Train a fresh foundation model with given hidden_ch."""
    all_x, all_y = [], []
    for tn, fn in ALL_TASKS.items():
        x, y = fn(N_PER_TASK, GS, seed=SEED); all_x.append(x); all_y.append(y)
    x_train, y_train = torch.cat(all_x), torch.cat(all_y)
    n = x_train.size(0)
    model = PromptLNCA(hc=hc).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(TRAIN_EPOCHS):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            xb, yb = x_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (epoch+1)%20==0: print(f"      Epoch {epoch+1}")
    return model


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase91_compute_scaling.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment':'Phase 91: Compute Scaling','timestamp':datetime.now().isoformat(),
                   'results':results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("="*70); print("Phase 91: TTT & Inference Compute Scaling")
    print(f"  Using hidden_ch={BEST_HC}"); print("="*70)

    # Train foundation
    print("\n  Training foundation model...")
    backbone = train_foundation(BEST_HC)
    backbone_state = copy.deepcopy(backbone.state_dict())

    # Prepare test data
    test_data = {}
    for tn, fn in ALL_TASKS.items():
        x, y = fn(N_TEST + N_DEMO + 10, GS, seed=SEED+20)
        test_data[tn] = {'demo_x':x[:N_DEMO].to(DEVICE), 'demo_y':y[:N_DEMO].to(DEVICE),
                         'test_x':x[N_DEMO+10:], 'test_y':y[N_DEMO+10:]}

    results = {}

    # A) Scale TTT steps (fixed T=10)
    print("\n  --- A) TTT Step Scaling (T=10) ---")
    ttt_steps_list = [0, 15, 30, 50, 80]
    results['ttt_scaling'] = {}
    for tn in ALL_TASKS:
        print(f"    Task: {tn}")
        td = test_data[tn]
        task_res = []
        for ttt_s in ttt_steps_list:
            model = PromptLNCA(hc=BEST_HC).to(DEVICE)
            model.load_state_dict(backbone_state); model.freeze_backbone()
            model.task_context = nn.Parameter(torch.zeros(1,CTX_CH,1,1,device=DEVICE))
            if ttt_s > 0:
                opt = torch.optim.Adam([model.task_context], lr=TTT_LR)
                t0 = time.perf_counter()
                for _ in range(ttt_s):
                    opt.zero_grad(); F.cross_entropy(model(td['demo_x']),td['demo_y']).backward(); opt.step()
                lat = (time.perf_counter()-t0)*1000
            else: lat = 0
            model.eval()
            with torch.no_grad():
                p = model(td['test_x'].to(DEVICE)).argmax(1); t = td['test_y'].to(DEVICE)
                px = (p==t).float().mean().item()
                ex = (p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
            task_res.append({'ttt_steps':ttt_s,'latency_ms':lat,'pixel':px,'exact':ex})
            print(f"      ttt={ttt_s:3d}: {lat:6.0f}ms pixel={px*100:.1f}% exact={ex*100:.1f}%")
            del model; gc.collect()
        results['ttt_scaling'][tn] = task_res
        _save(results)

    # B) Scale inference T (fixed TTT=30)
    print("\n  --- B) NCA Steps Scaling (TTT=30) ---")
    nca_steps_list = [5, 10, 15, 20, 30, 50]
    results['nca_scaling'] = {}
    for tn in ALL_TASKS:
        print(f"    Task: {tn}")
        td = test_data[tn]
        task_res = []
        for T in nca_steps_list:
            model = PromptLNCA(hc=BEST_HC).to(DEVICE)
            model.load_state_dict(backbone_state); model.freeze_backbone()
            model.task_context = nn.Parameter(torch.zeros(1,CTX_CH,1,1,device=DEVICE))
            opt = torch.optim.Adam([model.task_context], lr=TTT_LR)
            for _ in range(30):
                opt.zero_grad(); F.cross_entropy(model(td['demo_x'], n_steps=T),td['demo_y']).backward(); opt.step()
            model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                p = model(td['test_x'].to(DEVICE), n_steps=T).argmax(1); t = td['test_y'].to(DEVICE)
                px = (p==t).float().mean().item()
                ex = (p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
            inf_lat = (time.perf_counter()-t0)*1000
            task_res.append({'nca_steps':T,'inf_latency_ms':inf_lat,'pixel':px,'exact':ex})
            print(f"      T={T:3d}: inf={inf_lat:6.0f}ms pixel={px*100:.1f}% exact={ex*100:.1f}%")
            del model; gc.collect()
        results['nca_scaling'][tn] = task_res
        _save(results)

    print(f"\n{'='*70}"); print("GRAND SUMMARY"); print(f"{'='*70}")
    for section in ['ttt_scaling','nca_scaling']:
        print(f"  {section}:")
        for tn, entries in results[section].items():
            best = max(entries, key=lambda e: e['exact'])
            key = 'ttt_steps' if 'ttt_steps' in best else 'nca_steps'
            print(f"    {tn}: best exact={best['exact']*100:.1f}% at {key}={best[key]}")

    _gen_fig(results); print("\nPhase 91 complete!")

def _gen_fig(results):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        for i, (section,xlabel) in enumerate([('ttt_scaling','TTT Steps'),('nca_scaling','NCA Steps (T)')]):
            ax = axes[i]
            for tn, entries in results[section].items():
                key = 'ttt_steps' if 'ttt_steps' in entries[0] else 'nca_steps'
                xs = [e[key] for e in entries]; ys = [e['exact']*100 for e in entries]
                ax.plot(xs, ys, 'o-', label=tn, linewidth=2, markersize=7)
            ax.set_xlabel(xlabel); ax.set_ylabel('Exact Match (%)'); ax.legend(); ax.grid(alpha=0.3)
            ax.set_title(f'{xlabel} Scaling', fontweight='bold')
        fig.suptitle('Phase 91: Compute Scaling Law',fontsize=13,fontweight='bold')
        plt.tight_layout(); os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR,"phase91_compute_scaling.png"),bbox_inches='tight',dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e: print(f"  Figure error: {e}")

if __name__ == '__main__': main()
