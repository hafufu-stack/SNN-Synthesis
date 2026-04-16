"""
Phase 95: Attractor Regularization

Train L-NCA with random T (5-20) + attractor loss:
  L_total = L_task + lambda * ||state(T) - state(T-1)||^2

Goal: survive T=30+ without collapse.

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
DEVICE = "cpu"
SEED = 2026
NC=10; HC=32; GS=8; BS=32; EPOCHS=60; LR=1e-3
N_TRAIN=1000; N_TEST=200
T_MIN=5; T_MAX=20; LAMBDA_ATT=0.1
TEST_STEPS = [5, 10, 15, 20, 30, 50]

def oh(g):
    h,w=g.shape; o=np.zeros((NC,h,w),dtype=np.float32)
    for c in range(NC): o[c]=(g==c).astype(np.float32)
    return o

def gen_expand(n,gs=8,seed=None):
    rng=np.random.RandomState(seed); ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,4)):
            y,x=rng.randint(1,gs-1),rng.randint(1,gs-1)
            if g[y,x]==0: g[y,x]=rng.randint(1,5)
        res=g.copy()
        for y in range(gs):
            for x in range(gs):
                if g[y,x]>0:
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny,nx=y+dy,x+dx
                        if 0<=ny<gs and 0<=nx<gs and res[ny,nx]==0: res[ny,nx]=g[y,x]
        ins.append(oh(g)); tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

def gen_gravity(n,gs=8,seed=None):
    rng=np.random.RandomState(seed); ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(0,gs),rng.randint(0,gs)]=1
        for _ in range(rng.randint(1,3)):
            r,c=rng.randint(0,gs),rng.randint(0,gs)
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
        ins.append(oh(g)); tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))


class AttractorLNCA(nn.Module):
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.hc=hc
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)

    def forward(self, x, n_steps=10, return_states=False):
        b,c,h,w=x.shape; state=torch.zeros(b,self.hc,h,w,device=x.device)
        states = []
        for _ in range(n_steps):
            combined=torch.cat([x,state],1); delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
            if return_states: states.append(state)
        out = self.readout(state)
        if return_states: return out, states
        return out


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase95_attractor.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 95: Attractor Regularization','timestamp':datetime.now().isoformat(),
                   'results':results},f,indent=2,default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("="*70); print("Phase 95: Attractor Regularization"); print("="*70)

    TASKS = {'gravity':gen_gravity, 'expand':gen_expand}
    results = {}

    for task_name, gen_fn in TASKS.items():
        print(f"\n  === Task: {task_name} ===")
        x_train, y_train = gen_fn(N_TRAIN, GS, seed=SEED)
        x_test, y_test = gen_fn(N_TEST, GS, seed=SEED+1)

        # A) Baseline (fixed T=10)
        print("    [Baseline] Fixed T=10, no regularization")
        model_base = AttractorLNCA().to(DEVICE)
        opt = torch.optim.Adam(model_base.parameters(), lr=LR)
        for ep in range(EPOCHS):
            model_base.train(); perm=torch.randperm(N_TRAIN)
            for i in range(0,N_TRAIN,BS):
                idx=perm[i:i+BS]; xb,yb=x_train[idx].to(DEVICE),y_train[idx].to(DEVICE)
                opt.zero_grad(); F.cross_entropy(model_base(xb,10),yb).backward()
                torch.nn.utils.clip_grad_norm_(model_base.parameters(),1.0); opt.step()

        # B) Attractor-regularized (random T + L2 penalty)
        print("    [Attractor] Random T=5-20 + L2 state penalty")
        model_att = AttractorLNCA().to(DEVICE)
        opt = torch.optim.Adam(model_att.parameters(), lr=LR)
        for ep in range(EPOCHS):
            model_att.train(); perm=torch.randperm(N_TRAIN)
            for i in range(0,N_TRAIN,BS):
                idx=perm[i:i+BS]; xb,yb=x_train[idx].to(DEVICE),y_train[idx].to(DEVICE)
                T = random.randint(T_MIN, T_MAX)
                opt.zero_grad()
                out, states = model_att(xb, T, return_states=True)
                loss_task = F.cross_entropy(out, yb)
                # Attractor penalty: last 2 states should be close
                if len(states) >= 2:
                    loss_att = ((states[-1] - states[-2])**2).mean()
                else:
                    loss_att = torch.tensor(0.0)
                loss = loss_task + LAMBDA_ATT * loss_att
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_att.parameters(),1.0); opt.step()

        # Evaluate both at various T
        task_results = {'baseline':[], 'attractor':[]}
        for T in TEST_STEPS:
            for label, model in [('baseline',model_base),('attractor',model_att)]:
                model.eval()
                with torch.no_grad():
                    p=model(x_test.to(DEVICE),T).argmax(1); t=y_test.to(DEVICE)
                    px=(p==t).float().mean().item()
                    ex=(p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
                task_results[label].append({'T':T,'pixel':px,'exact':ex})
                print(f"      {label:10s} T={T:3d}: px={px*100:.1f}% ex={ex*100:.1f}%")

        results[task_name] = task_results
        _save(results)
        del model_base, model_att; gc.collect()

    print(f"\n{'='*70}"); print("GRAND SUMMARY: Attractor Regularization"); print(f"{'='*70}")
    for tn, tr in results.items():
        print(f"  {tn}:")
        for label in ['baseline','attractor']:
            entries = tr[label]
            for e in entries:
                marker = " <--" if e['T']==30 else ""
                print(f"    {label:10s} T={e['T']:3d}: px={e['pixel']*100:.1f}% ex={e['exact']*100:.1f}%{marker}")

    _gen_fig(results); print("\nPhase 95 complete!")


def _gen_fig(results):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        tasks = list(results.keys())
        fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 5))
        if len(tasks)==1: axes=[axes]
        for i, tn in enumerate(tasks):
            ax=axes[i]
            for label, color in [('baseline','#9CA3AF'),('attractor','#EC4899')]:
                entries = results[tn][label]
                Ts = [e['T'] for e in entries]
                exacts = [e['exact']*100 for e in entries]
                ax.plot(Ts, exacts, 'o-', color=color, label=label, linewidth=2, markersize=7)
            ax.set_xlabel('NCA Steps (T)'); ax.set_ylabel('Exact Match (%)')
            ax.set_title(f'{tn}', fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
            ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
        fig.suptitle('Phase 95: Attractor Regularization\nBaseline vs Attractor-Reg at varying T',
                    fontsize=13, fontweight='bold')
        plt.tight_layout(); os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR,"phase95_attractor.png"),bbox_inches='tight',dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e: print(f"  Figure error: {e}")

if __name__ == '__main__': main()
