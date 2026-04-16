"""
Phase 93: Massive L-MoE Library (20 specialists)

Can Zero-shot routing stay 100% accurate with 20 experts?

20 ARC-like tasks: gravity, expand, color_invert, move_right, fill_border,
rotate_90, flip_h, flip_v, color_replace, denoise, flood_fill, shrink,
move_up, move_left, move_down, diagonal_move, checkerboard, outline,
color_sort, mirror_diag

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
DEVICE = "cpu"
SEED = 2026
NC = 10; HC = 32; NCA_STEPS = 10; GS = 8; BS = 32
SPECIALIST_EPOCHS = 40; N_TRAIN = 800; N_TEST = 100; N_DEMO = 3
CTX_CH = 4; TTT_STEPS = 30; TTT_LR = 0.1; TIME_BUDGET = 500

def oh(g):
    h,w=g.shape; o=np.zeros((NC,h,w),dtype=np.float32)
    for c in range(NC): o[c]=(g==c).astype(np.float32)
    return o

# === 20 Task Generators ===
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

def _color_invert(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,7)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        res=g.copy(); res[g==1]=2; res[g==2]=1
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

def _fill_border(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        res=g.copy(); res[0,:]=3; res[-1,:]=3; res[:,0]=3; res[:,-1]=3
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _move_up(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0: res[(r-1)%gs,c]=g[r,c]
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _move_left(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0: res[r,(c-1)%gs]=g[r,c]
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _move_down(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0: res[(r+1)%gs,c]=g[r,c]
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _flip_h(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,6)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        ins.append(oh(g)); tgs.append(np.fliplr(g).copy())
    return ins, tgs

def _flip_v(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,6)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        ins.append(oh(g)); tgs.append(np.flipud(g).copy())
    return ins, tgs

def _color_replace(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,8)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        res=g.copy(); res[g==3]=4; res[g==4]=3
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _denoise(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        col=rng.randint(1,5)
        for _ in range(rng.randint(4,10)): g[rng.randint(0,gs),rng.randint(0,gs)]=col
        noisy=g.copy()
        for _ in range(rng.randint(1,3)): noisy[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        ins.append(oh(noisy)); tgs.append(g)
    return ins, tgs

def _outline(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(1,gs-1),rng.randint(1,gs-1)]=rng.randint(1,4)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0:
                    res[r,c]=g[r,c]
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc_=r+dr,c+dc
                        if 0<=nr<gs and 0<=nc_<gs and res[nr,nc_]==0 and g[nr,nc_]==0:
                            res[nr,nc_]=5
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _checkerboard(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        c1,c2=rng.randint(1,4),rng.randint(4,7)
        for _ in range(rng.randint(1,3)): g[rng.randint(0,gs),rng.randint(0,gs)]=c1
        res=g.copy()
        for r in range(gs):
            for c in range(gs):
                if g[r,c]==c1: res[r,c]=c1 if (r+c)%2==0 else c2
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _diagonal(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0: res[(r+1)%gs,(c+1)%gs]=g[r,c]
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _color_sort(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,7)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        res=g.copy()
        for c in range(gs):
            col_vals=[g[r,c] for r in range(gs) if g[r,c]>0]
            col_vals.sort()
            idx=0
            for r in range(gs):
                if g[r,c]>0: res[r,c]=col_vals[idx]; idx+=1
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _mirror_diag(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,6)): g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        res=g.T.copy()
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _shrink(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,8)):
            r,c=rng.randint(1,gs-1),rng.randint(1,gs-1)
            g[r,c]=rng.randint(1,4)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc_=r+dr,c+dc
                if 0<=nr<gs and 0<=nc_<gs: g[nr,nc_]=g[r,c]
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0:
                    ct=0
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc_=r+dr,c+dc
                        if 0<=nr<gs and 0<=nc_<gs and g[nr,nc_]==g[r,c]: ct+=1
                    if ct>=2: res[r,c]=g[r,c]
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def _flood(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        sr,sc=rng.randint(1,gs-1),rng.randint(1,gs-1)
        g[sr,sc]=rng.randint(1,4)
        res=g.copy()
        stack=[(sr,sc)]
        visited=set()
        while stack:
            r,c=stack.pop()
            if (r,c) in visited: continue
            visited.add((r,c))
            res[r,c]=g[sr,sc]
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc_=r+dr,c+dc
                if 0<=nr<gs and 0<=nc_<gs and (nr,nc_) not in visited: stack.append((nr,nc_))
        ins.append(oh(g)); tgs.append(res)
    return ins, tgs

def gen_task(name, n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    TASKS = {
        'gravity':_gravity,'expand':_expand,'color_invert':_color_invert,
        'move_right':_move_right,'fill_border':_fill_border,'move_up':_move_up,
        'move_left':_move_left,'move_down':_move_down,'flip_h':_flip_h,
        'flip_v':_flip_v,'color_replace':_color_replace,'denoise':_denoise,
        'outline':_outline,'checkerboard':_checkerboard,'diagonal':_diagonal,
        'color_sort':_color_sort,'mirror_diag':_mirror_diag,'shrink':_shrink,
        'flood':_flood,
    }
    ins, tgs = TASKS[name](n, gs, rng)
    return torch.tensor(np.array(ins)), torch.tensor(np.array(tgs))

ALL_TASK_NAMES = ['gravity','expand','color_invert','move_right','fill_border',
                  'move_up','move_left','move_down','flip_h','flip_v',
                  'color_replace','denoise','outline','checkerboard','diagonal',
                  'color_sort','mirror_diag','shrink','flood']


class LNCA(nn.Module):
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.hc=hc
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
        self.task_context=nn.Parameter(torch.zeros(1,CTX_CH,1,1))
        self.ctx_proj=nn.Conv2d(CTX_CH,nc,1,bias=False)
    def forward(self, x, n_steps=NCA_STEPS):
        b,c,h,w=x.shape
        ctx=self.ctx_proj(self.task_context.expand(b,-1,h,w)); x_a=x+ctx
        state=torch.zeros(b,self.hc,h,w,device=x.device)
        for _ in range(n_steps):
            combined=torch.cat([x_a,state],1); delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x_a,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
        return self.readout(state)
    def freeze_backbone(self):
        for name,p in self.named_parameters():
            if name!='task_context': p.requires_grad=False


def train_specialist(task_name):
    x,y = gen_task(task_name, N_TRAIN, GS, seed=SEED)
    model = LNCA().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(SPECIALIST_EPOCHS):
        model.train(); perm=torch.randperm(N_TRAIN)
        for i in range(0,N_TRAIN,BS):
            idx=perm[i:i+BS]; xb,yb=x[idx].to(DEVICE),y[idx].to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    with torch.no_grad():
        p=model(x[:100].to(DEVICE)).argmax(1)
        acc=(p==y[:100].to(DEVICE)).float().mean().item()
    return model.state_dict(), acc


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase93_massive_moe.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 93: Massive L-MoE','timestamp':datetime.now().isoformat(),
                   'results':results},f,indent=2,default=str)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("="*70); print(f"Phase 93: Massive L-MoE ({len(ALL_TASK_NAMES)} specialists)"); print("="*70)

    # Train all specialists
    specialists = {}
    results = {'specialist_acc':{}}
    for tn in ALL_TASK_NAMES:
        state, acc = train_specialist(tn)
        specialists[tn] = state
        results['specialist_acc'][tn] = acc
        print(f"  {tn:15s}: train pixel={acc*100:.1f}%")

    # MoE test
    print(f"\n  Running MoE on {len(ALL_TASK_NAMES)} mixed levels...")
    rng = np.random.RandomState(SEED+700)
    total_solved, correct_routes, total = 0, 0, 0
    results['levels'] = []

    for level, tn in enumerate(ALL_TASK_NAMES):
        gs = [8,10,12][level%3]
        x,y = gen_task(tn, N_DEMO+1, gs, seed=SEED+800+level)
        dx,dy = x[:N_DEMO].to(DEVICE),y[:N_DEMO].to(DEVICE)
        tx,ty = x[N_DEMO:N_DEMO+1],y[N_DEMO:N_DEMO+1]

        t0=time.perf_counter()
        best_loss,best_tn=float('inf'),None
        for stn,state in specialists.items():
            m=LNCA().to(DEVICE); m.load_state_dict(state); m.eval()
            with torch.no_grad(): loss=F.cross_entropy(m(dx),dy).item()
            if loss<best_loss: best_loss=loss; best_tn=stn
            del m

        m=LNCA().to(DEVICE); m.load_state_dict(specialists[best_tn])
        m.freeze_backbone()
        m.task_context=nn.Parameter(torch.zeros(1,CTX_CH,1,1,device=DEVICE))
        opt=torch.optim.Adam([m.task_context],lr=TTT_LR)
        for _ in range(TTT_STEPS):
            opt.zero_grad(); F.cross_entropy(m(dx),dy).backward(); opt.step()

        m.eval()
        with torch.no_grad():
            p=m(tx.to(DEVICE)).argmax(1); t=ty.to(DEVICE)
            px=(p==t).float().mean().item()
            ex=(p.reshape(1,-1)==t.reshape(1,-1)).all(1).float().mean().item()
        ms=(time.perf_counter()-t0)*1000
        solved=ex>0.5; route_ok=(best_tn==tn)
        total_solved+=int(solved); correct_routes+=int(route_ok); total+=1

        status="SOLVED" if solved else "WRONG"
        rm="OK" if route_ok else "MISS"
        print(f"  Lv{level:2d} ({tn:>14s} {gs}x{gs}): {status:6s} "
              f"route={best_tn}({rm}) px={px*100:.0f}% {ms:.0f}ms")
        results['levels'].append({'level':level,'task':tn,'grid_size':gs,
            'routed_to':best_tn,'route_correct':route_ok,
            'pixel':px,'exact':ex,'ms':ms})
        del m; gc.collect()

    sr=total_solved/total; ra=correct_routes/total
    results['summary']={'solve_rate':sr,'route_accuracy':ra,
        'solved':total_solved,'total':total,'avg_ms':np.mean([l['ms'] for l in results['levels']])}
    _save(results)

    print(f"\n{'='*70}"); print("GRAND SUMMARY: Massive L-MoE"); print(f"{'='*70}")
    print(f"  Solve Rate: {total_solved}/{total} = {sr*100:.0f}%")
    print(f"  Route Accuracy: {ra*100:.0f}%")
    print(f"  Avg Latency: {results['summary']['avg_ms']:.0f}ms")
    print("\nPhase 93 complete!")

if __name__ == '__main__': main()
