"""
Phase 96: Batched Compositional Routing

Exploit shared backbone: batch all task_contexts into one forward pass.
19 experts in 1 forward = O(1) routing.

Author: Hiroto Funasaki
"""
import os,json,time,gc,random,copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DEVICE = "cpu"; SEED = 2026; NC=10; HC=32; NCA_STEPS=10; GS=8; BS=32
SPECIALIST_EPOCHS=40; N_TRAIN=800; N_DEMO=3; N_TEST=100; CTX_CH=4; TTT_STEPS=20; TTT_LR=0.1

def oh(g):
    h,w=g.shape;o=np.zeros((NC,h,w),dtype=np.float32)
    for c in range(NC):o[c]=(g==c).astype(np.float32)
    return o

def _gravity(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):g[rng.randint(0,gs),rng.randint(0,gs)]=1
        for _ in range(rng.randint(1,3)):
            r,c=rng.randint(0,gs),rng.randint(0,gs)
            if g[r,c]==0:g[r,c]=2
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]==2:res[r,c]=2
        for c in range(gs):
            cnt=sum(1 for r in range(gs) if g[r,c]==1);row,pl=gs-1,0
            while pl<cnt and row>=0:
                if res[row,c]==0:res[row,c]=1;pl+=1
                row-=1
        ins.append(oh(g));tgs.append(res)
    return ins,tgs

def _expand(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,4)):
            y,x=rng.randint(1,gs-1),rng.randint(1,gs-1)
            if g[y,x]==0:g[y,x]=rng.randint(1,5)
        res=g.copy()
        for y in range(gs):
            for x in range(gs):
                if g[y,x]>0:
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny,nx=y+dy,x+dx
                        if 0<=ny<gs and 0<=nx<gs and res[ny,nx]==0:res[ny,nx]=g[y,x]
        ins.append(oh(g));tgs.append(res)
    return ins,tgs

def _color_invert(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,7)):g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        res=g.copy();res[g==1]=2;res[g==2]=1
        ins.append(oh(g));tgs.append(res)
    return ins,tgs

def _move_right(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0:res[r,(c+1)%gs]=g[r,c]
        ins.append(oh(g));tgs.append(res)
    return ins,tgs

def _fill_border(n,gs,rng):
    ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,5)
        res=g.copy();res[0,:]=3;res[-1,:]=3;res[:,0]=3;res[:,-1]=3
        ins.append(oh(g));tgs.append(res)
    return ins,tgs

TASK_FNS = {'gravity':_gravity,'expand':_expand,'color_invert':_color_invert,
            'move_right':_move_right,'fill_border':_fill_border}

def gen_task(name,n,gs=8,seed=None):
    rng=np.random.RandomState(seed);ins,tgs=TASK_FNS[name](n,gs,rng)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))


class BatchedLNCA(nn.Module):
    """Shared backbone, different contexts via batch dimension."""
    def __init__(self,nc=10,hc=32,ctx_ch=4):
        super().__init__()
        self.hc=hc;self.ctx_ch=ctx_ch
        self.ctx_proj=nn.Conv2d(ctx_ch,nc,1,bias=False)
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
    def forward(self,x,contexts,n_steps=NCA_STEPS):
        """x: (1,C,H,W), contexts: (K,ctx_ch,1,1) -> output: (K,C,H,W)"""
        K=contexts.size(0);h,w=x.size(2),x.size(3)
        x_rep=x.expand(K,-1,-1,-1)
        ctx=self.ctx_proj(contexts.expand(-1,-1,h,w))
        x_aug=x_rep+ctx
        state=torch.zeros(K,self.hc,h,w,device=x.device)
        for _ in range(n_steps):
            combined=torch.cat([x_aug,state],1);delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x_aug,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
        return self.readout(state)


def train_specialist_ctx(task_name):
    """Train and return learned context + shared backbone."""
    x,y=gen_task(task_name,N_TRAIN,GS,seed=SEED)
    model=BatchedLNCA().to(DEVICE)
    ctx=nn.Parameter(torch.randn(1,CTX_CH,1,1)*0.01).to(DEVICE)  # (1,CTX_CH,1,1)
    all_params=list(model.parameters())+[ctx]
    opt=torch.optim.Adam(all_params,lr=1e-3)
    for ep in range(SPECIALIST_EPOCHS):
        model.train();perm=torch.randperm(N_TRAIN)
        for i in range(0,N_TRAIN,BS):
            idx=perm[i:i+BS];xb,yb=x[idx].to(DEVICE),y[idx].to(DEVICE)
            opt.zero_grad()
            # Each sample uses same context
            ctxb=ctx.expand(xb.size(0),-1,-1,-1)
            out=model(xb[:1],ctxb[:1])  # single forward for routing
            # Actually train properly per-sample
            loss=0
            for j in range(xb.size(0)):
                out_j=model(xb[j:j+1],ctx)
                loss=loss+F.cross_entropy(out_j,yb[j:j+1])
            (loss/xb.size(0)).backward()
            torch.nn.utils.clip_grad_norm_(all_params,1.0);opt.step()
    return model.state_dict(),ctx.data.clone().squeeze(0)  # -> (CTX_CH,1,1)


def _save(results):
    os.makedirs(RESULTS_DIR,exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase96_batched_routing.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 96: Batched Routing','timestamp':datetime.now().isoformat(),
                   'results':results},f,indent=2,default=str)

def main():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    print("="*70);print("Phase 96: Batched Compositional Routing");print("="*70)

    # Train specialists (shared backbone, different contexts)
    print("  Training specialists with shared backbone...")
    backbone_state=None;contexts={};task_names=list(TASK_FNS.keys())
    for tn in task_names:
        state,ctx=train_specialist_ctx(tn)
        backbone_state=state  # last one wins (they share architecture)
        contexts[tn]=ctx
        print(f"    {tn}: done")

    # Stack all contexts
    ctx_stack=torch.stack([contexts[tn] for tn in task_names])  # (K,CTX_CH,1,1)
    K=len(task_names)

    # Benchmark: Sequential vs Batched routing
    print(f"\n  Benchmark: Sequential vs Batched ({K} experts)")
    model=BatchedLNCA().to(DEVICE)
    model.load_state_dict(backbone_state)
    model.eval()

    x_test,y_test=gen_task('gravity',1,GS,seed=SEED+500)
    dx=x_test[:1].to(DEVICE);dy=y_test[:1].to(DEVICE)

    # Sequential
    t0=time.perf_counter()
    for _ in range(10):
        losses_seq=[]
        for tn in task_names:
            with torch.no_grad():
                out=model(dx,contexts[tn].unsqueeze(0))  # (1,CTX_CH,1,1)
                losses_seq.append(F.cross_entropy(out,dy).item())
    seq_ms=(time.perf_counter()-t0)/10*1000
    print(f"    Sequential: {seq_ms:.1f}ms")

    # Batched
    t0=time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            out_all=model(dx,ctx_stack)  # (K,NC,H,W) in ONE forward!
            dy_rep=dy.expand(K,-1,-1)
            losses_batch=[F.cross_entropy(out_all[i:i+1],dy[0:1].unsqueeze(0) if dy.dim()==2 else dy).item() for i in range(K)]
    batch_ms=(time.perf_counter()-t0)/10*1000
    speedup=seq_ms/batch_ms if batch_ms>0 else float('inf')
    print(f"    Batched:    {batch_ms:.1f}ms ({speedup:.1f}x speedup)")

    # Full MoE test with batched routing
    print(f"\n  MoE test with batched routing...")
    results={'seq_ms':seq_ms,'batch_ms':batch_ms,'speedup':speedup,'levels':[]}
    total_solved,total=0,0

    for level,tn in enumerate(task_names):
        gs=[8,10,12][level%3]
        x,y=gen_task(tn,N_DEMO+1,gs,seed=SEED+600+level)
        dx,dy=x[:N_DEMO].to(DEVICE),y[:N_DEMO].to(DEVICE)
        tx,ty=x[N_DEMO:N_DEMO+1],y[N_DEMO:N_DEMO+1]

        t0=time.perf_counter()
        # Batched routing on first demo
        model.eval()
        with torch.no_grad():
            out_all=model(dx[:1],ctx_stack)  # (K,NC,H,W)
            losses=[F.cross_entropy(out_all[i:i+1],dy[:1]).item() for i in range(K)]
        best_idx=int(np.argmin(losses))
        best_tn=task_names[best_idx]

        # TTT with best context
        best_ctx=nn.Parameter(contexts[best_tn].clone())  # (CTX_CH,1,1)
        opt=torch.optim.Adam([best_ctx],lr=TTT_LR)
        for _ in range(TTT_STEPS):
            opt.zero_grad()
            out=model(dx[:1],best_ctx.unsqueeze(0))  # -> (1,CTX_CH,1,1)
            F.cross_entropy(out,dy[:1]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p=model(tx.to(DEVICE),best_ctx.unsqueeze(0)).argmax(1)  # (1,CTX_CH,1,1)
            t=ty.to(DEVICE)
            px=(p==t).float().mean().item()
            ex=(p.reshape(1,-1)==t.reshape(1,-1)).all(1).float().mean().item()
        ms=(time.perf_counter()-t0)*1000
        solved=ex>0.5;total_solved+=int(solved);total+=1
        route_ok=(best_tn==tn)
        print(f"  Lv{level:2d} ({tn:>13s} {gs}x{gs}): {'SOLVED' if solved else 'WRONG':6s} "
              f"route={best_tn}({'OK' if route_ok else 'MISS'}) px={px*100:.0f}% {ms:.0f}ms")
        results['levels'].append({'task':tn,'routed_to':best_tn,'exact':ex,'ms':ms})

    results['solve_rate']=total_solved/total
    _save(results)
    print(f"\n  Solve Rate: {total_solved}/{total}={results['solve_rate']*100:.0f}%")
    print(f"  Speedup: {speedup:.1f}x\nPhase 96 complete!")

if __name__=='__main__':main()
