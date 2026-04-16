"""
Phase 94: Compositional Routing (2-step Expert Chaining)

Test: can we solve composite tasks (e.g. color_invert + gravity)
by chaining Expert_B(Expert_A(x)) and finding the best pair via
exhaustive Zero-shot Loss search?

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy, itertools
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
NC = 10; HC = 32; NCA_STEPS = 10; GS = 8; BS = 32
SPECIALIST_EPOCHS = 40; N_TRAIN = 800; N_TEST = 100; N_DEMO = 3
CTX_CH = 4; TTT_STEPS = 20; TTT_LR = 0.1

def oh(g):
    h,w=g.shape; o=np.zeros((NC,h,w),dtype=np.float32)
    for c in range(NC): o[c]=(g==c).astype(np.float32)
    return o

# Reuse task generators from phase93
def _color_invert(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,7)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        res=g.copy(); res[g==1]=2; res[g==2]=1
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _gravity(n,gs,rng):
    ins,tgs=[],[]
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
    return ins, tgs

def _move_right(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0: res[r,(c+1)%gs]=g[r,c]
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _expand(n,gs,rng):
    ins,tgs=[],[]
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
    return ins, tgs

SINGLE_TASKS = {
    'color_invert': _color_invert,
    'gravity': _gravity,
    'move_right': _move_right,
    'expand': _expand,
}

# Composite task generators: apply A then B
def gen_composite(task_a_fn, task_b_fn, n, gs=8, seed=None):
    """Generate composite: B(A(input))"""
    rng = np.random.RandomState(seed)
    # Generate inputs, apply A, then apply B
    ins_a, tgs_a = task_a_fn(n, gs, rng)
    ins, tgs = [], []
    for i in range(n):
        # Input is original
        inp = ins_a[i]  # one-hot
        mid = tgs_a[i]  # after task A (class indices)
        # Apply task B to the intermediate result
        rng2 = np.random.RandomState(seed + 1000 + i)
        mid_oh_list, final_list = task_b_fn(1, gs, rng2)
        # Actually, we need B applied to mid, not random input
        # So we apply B's transformation to mid directly
        # Simpler: just compose via numpy
        ins.append(inp)
        tgs.append(mid)  # placeholder, will fix below
    return torch.tensor(np.array(ins)), torch.tensor(np.array([t for t in tgs_a]))


def apply_gravity_to_grid(g):
    gs=g.shape[0]; res=np.zeros_like(g)
    for r in range(gs):
        for c in range(gs):
            if g[r,c]==2: res[r,c]=2
    for c in range(gs):
        cnt=sum(1 for r in range(gs) if g[r,c]==1); row,pl=gs-1,0
        while pl<cnt and row>=0:
            if res[row,c]==0: res[row,c]=1; pl+=1
            row-=1
    return res

def apply_color_invert(g):
    res=g.copy(); res[g==1]=2; res[g==2]=1; return res

def apply_move_right(g):
    gs=g.shape[0]; res=np.zeros_like(g)
    for r in range(gs):
        for c in range(gs):
            if g[r,c]>0: res[r,(c+1)%gs]=g[r,c]
    return res


def gen_composite_direct(task_a_name, task_b_name, n, gs=8, seed=None):
    """Generate B(A(x)) composite tasks."""
    rng = np.random.RandomState(seed)
    apply_fns = {
        'color_invert': apply_color_invert,
        'gravity': apply_gravity_to_grid,
        'move_right': apply_move_right,
    }
    ins, tgs = [], []
    _, raw_inputs = SINGLE_TASKS[task_a_name](n, gs, rng)
    rng2 = np.random.RandomState(seed)
    raw_a, _ = SINGLE_TASKS[task_a_name](n, gs, rng2)

    rng3 = np.random.RandomState(seed)
    for i in range(n):
        g = np.zeros((gs,gs), dtype=np.int64)
        for _ in range(rng3.randint(2,5)):
            g[rng3.randint(0,gs), rng3.randint(0,gs)] = rng3.randint(1,4)
        mid = apply_fns[task_a_name](g)
        final = apply_fns[task_b_name](mid)
        ins.append(oh(g)); tgs.append(final)
    return torch.tensor(np.array(ins)), torch.tensor(np.array(tgs))


class LNCA(nn.Module):
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.hc=hc
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
    def forward(self, x, n_steps=NCA_STEPS):
        b,c,h,w=x.shape; state=torch.zeros(b,self.hc,h,w,device=x.device)
        for _ in range(n_steps):
            combined=torch.cat([x,state],1); delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
        return self.readout(state)


def train_specialist(task_name):
    rng=np.random.RandomState(SEED)
    ins,tgs = SINGLE_TASKS[task_name](N_TRAIN, GS, rng)
    x=torch.tensor(np.array(ins)); y=torch.tensor(np.array(tgs))
    model=LNCA().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(SPECIALIST_EPOCHS):
        model.train(); perm=torch.randperm(N_TRAIN)
        for i in range(0,N_TRAIN,BS):
            idx=perm[i:i+BS]; xb,yb=x[idx].to(DEVICE),y[idx].to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase94_compositional.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 94: Compositional Routing','timestamp':datetime.now().isoformat(),
                   'results':results},f,indent=2,default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("="*70); print("Phase 94: Compositional Routing (2-step Expert Chaining)"); print("="*70)

    # Train specialists
    print("  Training specialists...")
    specialists = {}
    for tn in SINGLE_TASKS:
        specialists[tn] = train_specialist(tn)
        print(f"    {tn}: done")

    # Define composite tests
    COMPOSITES = [
        ('color_invert', 'gravity'),
        ('color_invert', 'move_right'),
        ('move_right', 'color_invert'),
        ('gravity', 'color_invert'),
    ]

    results = {}
    for task_a, task_b in COMPOSITES:
        composite_name = f"{task_a}+{task_b}"
        print(f"\n  --- Composite: {composite_name} ---")

        x, y = gen_composite_direct(task_a, task_b, N_DEMO + N_TEST, GS, seed=SEED+900)
        dx, dy = x[:N_DEMO].to(DEVICE), y[:N_DEMO].to(DEVICE)
        tx, ty = x[N_DEMO:N_DEMO+N_TEST], y[N_DEMO:N_DEMO+N_TEST]

        # 1) Single expert baseline (best of all)
        best_loss_single, best_single = float('inf'), None
        for tn, model in specialists.items():
            model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(model(dx), dy).item()
            if loss < best_loss_single:
                best_loss_single = loss; best_single = tn

        model_s = specialists[best_single]
        model_s.eval()
        with torch.no_grad():
            p = model_s(tx.to(DEVICE)).argmax(1)
            px_s = (p == ty.to(DEVICE)).float().mean().item()
            ex_s = (p.reshape(N_TEST,-1)==ty.to(DEVICE).reshape(N_TEST,-1)).all(1).float().mean().item()
        print(f"    Single best ({best_single}): px={px_s*100:.1f}% ex={ex_s*100:.1f}%")

        # 2) 2-step chaining: try all pairs
        best_loss_pair, best_pair = float('inf'), None
        pair_results = []
        for tn_a, tn_b in itertools.product(specialists.keys(), repeat=2):
            ma, mb = specialists[tn_a], specialists[tn_b]
            ma.eval(); mb.eval()
            with torch.no_grad():
                mid = F.softmax(ma(dx), dim=1)  # soft output as input to B
                out = mb(mid)
                loss = F.cross_entropy(out, dy).item()
            pair_results.append((tn_a, tn_b, loss))
            if loss < best_loss_pair:
                best_loss_pair = loss; best_pair = (tn_a, tn_b)

        # Evaluate best pair
        ma, mb = specialists[best_pair[0]], specialists[best_pair[1]]
        ma.eval(); mb.eval()
        with torch.no_grad():
            mid = F.softmax(ma(tx.to(DEVICE)), dim=1)
            p = mb(mid).argmax(1)
            px_c = (p == ty.to(DEVICE)).float().mean().item()
            ex_c = (p.reshape(N_TEST,-1)==ty.to(DEVICE).reshape(N_TEST,-1)).all(1).float().mean().item()

        discovered = f"{best_pair[0]}+{best_pair[1]}"
        correct = (best_pair[0] == task_a and best_pair[1] == task_b)
        print(f"    2-step best ({discovered}): px={px_c*100:.1f}% ex={ex_c*100:.1f}% "
              f"{'CORRECT!' if correct else 'WRONG route'}")

        results[composite_name] = {
            'true_a': task_a, 'true_b': task_b,
            'single_best': best_single, 'single_px': px_s, 'single_ex': ex_s,
            'chain_best': list(best_pair), 'chain_px': px_c, 'chain_ex': ex_c,
            'route_correct': correct,
        }
        _save(results)

    print(f"\n{'='*70}"); print("GRAND SUMMARY: Compositional Routing"); print(f"{'='*70}")
    for comp, r in results.items():
        print(f"  {comp}: single={r['single_ex']*100:.0f}% chain={r['chain_ex']*100:.0f}% "
              f"found={r['chain_best'][0]}+{r['chain_best'][1]} "
              f"{'CORRECT' if r['route_correct'] else 'WRONG'}")
    print("\nPhase 94 complete!")

if __name__ == '__main__': main()
