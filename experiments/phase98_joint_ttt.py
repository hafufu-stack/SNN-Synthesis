"""
Phase 98: Joint Cellular Prompt Tuning for Composite Tasks

Chain Expert_A -> Expert_B and jointly TTT both contexts.
Goal: 88% -> 100% on composite tasks.

Author: Hiroto Funasaki
"""
import os,json,time,gc,random,copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings;warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
DEVICE="cpu";SEED=2026;NC=10;HC=32;NCA_STEPS=10;GS=8;BS=32
EPOCHS=50;N_TRAIN=1000;N_TEST=100;N_DEMO=3;CTX_CH=4;TTT_LR=0.05;TTT_STEPS=30;LR=1e-3

def oh(g):
    h,w=g.shape;o=np.zeros((NC,h,w),dtype=np.float32)
    for c in range(NC):o[c]=(g==c).astype(np.float32)
    return o

def apply_color_invert(g):
    res=g.copy();res[g==1]=2;res[g==2]=1;return res
def apply_gravity(g):
    gs=g.shape[0];res=np.zeros_like(g)
    for r in range(gs):
        for c in range(gs):
            if g[r,c]==2:res[r,c]=2
    for c in range(gs):
        cnt=sum(1 for r in range(gs) if g[r,c]==1);row,pl=gs-1,0
        while pl<cnt and row>=0:
            if res[row,c]==0:res[row,c]=1;pl+=1
            row-=1
    return res
def apply_move_right(g):
    gs=g.shape[0];res=np.zeros_like(g)
    for r in range(gs):
        for c in range(gs):
            if g[r,c]>0:res[r,(c+1)%gs]=g[r,c]
    return res

APPLY={'color_invert':apply_color_invert,'gravity':apply_gravity,'move_right':apply_move_right}

def gen_single(name,n,gs=8,seed=None):
    rng=np.random.RandomState(seed);ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        ins.append(oh(g));tgs.append(APPLY[name](g))
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

def gen_composite(a,b,n,gs=8,seed=None):
    rng=np.random.RandomState(seed);ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):g[rng.randint(0,gs),rng.randint(0,gs)]=rng.randint(1,4)
        mid=APPLY[a](g);final=APPLY[b](mid)
        ins.append(oh(g));tgs.append(final)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))


class PromptLNCA(nn.Module):
    def __init__(self,nc=10,hc=32,ctx_ch=4):
        super().__init__()
        self.hc=hc;self.ctx_proj=nn.Conv2d(ctx_ch,nc,1,bias=False)
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
    def forward(self,x,ctx,n_steps=NCA_STEPS):
        b,c,h,w=x.shape
        ctx_signal=self.ctx_proj(ctx.expand(b,-1,h,w));x_a=x+ctx_signal
        state=torch.zeros(b,self.hc,h,w,device=x.device)
        for _ in range(n_steps):
            combined=torch.cat([x_a,state],1);delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x_a,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
        return self.readout(state)


def train_specialist(name):
    x,y=gen_single(name,N_TRAIN,GS,seed=SEED)
    model=PromptLNCA().to(DEVICE)
    ctx=nn.Parameter(torch.zeros(1,CTX_CH,1,1))
    opt=torch.optim.Adam(list(model.parameters())+[ctx],lr=LR)
    for ep in range(EPOCHS):
        model.train();perm=torch.randperm(N_TRAIN)
        for i in range(0,N_TRAIN,BS):
            idx=perm[i:i+BS];xb,yb=x[idx].to(DEVICE),y[idx].to(DEVICE)
            opt.zero_grad();F.cross_entropy(model(xb,ctx),yb).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters())+[ctx],1.0);opt.step()
    return model,ctx.data.clone()

def _save(r):
    os.makedirs(RESULTS_DIR,exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase98_joint_ttt.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 98','timestamp':datetime.now().isoformat(),'results':r},f,indent=2,default=str)

def main():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    print("="*70);print("Phase 98: Joint Cellular Prompt Tuning");print("="*70)

    TASKS=['color_invert','gravity','move_right']
    specialists={};ctxs={}
    print("  Training specialists...")
    for tn in TASKS:
        m,c=train_specialist(tn);specialists[tn]=m;ctxs[tn]=c
        print(f"    {tn}: done")

    COMPOSITES=[('color_invert','gravity'),('gravity','color_invert'),('color_invert','move_right')]
    results={}

    for ta,tb in COMPOSITES:
        comp=f"{ta}+{tb}"
        print(f"\n  === Composite: {comp} ===")
        x,y=gen_composite(ta,tb,N_DEMO+N_TEST,GS,seed=SEED+900)
        dx,dy=x[:N_DEMO].to(DEVICE),y[:N_DEMO].to(DEVICE)
        tx,ty=x[N_DEMO:N_DEMO+N_TEST],y[N_DEMO:N_DEMO+N_TEST]

        ma,mb=specialists[ta],specialists[tb]

        # A) Chain without TTT
        ma.eval();mb.eval()
        with torch.no_grad():
            mid=F.softmax(ma(tx.to(DEVICE),ctxs[ta]),dim=1)
            p=mb(mid,ctxs[tb]).argmax(1);t=ty.to(DEVICE)
            px_no=(p==t).float().mean().item()
            ex_no=(p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
        print(f"    Chain (no TTT): pixel={px_no*100:.1f}% exact={ex_no*100:.1f}%")

        # B) Joint TTT: optimize ctx_A and ctx_B simultaneously
        ctx_a=nn.Parameter(ctxs[ta].clone())
        ctx_b=nn.Parameter(ctxs[tb].clone())
        opt=torch.optim.Adam([ctx_a,ctx_b],lr=TTT_LR)
        for step in range(TTT_STEPS):
            opt.zero_grad()
            mid=F.softmax(ma(dx,ctx_a),dim=1)
            out=mb(mid,ctx_b)
            loss=F.cross_entropy(out,dy)
            loss.backward()
            opt.step()

        ma.eval();mb.eval()
        with torch.no_grad():
            mid=F.softmax(ma(tx.to(DEVICE),ctx_a),dim=1)
            p=mb(mid,ctx_b).argmax(1);t=ty.to(DEVICE)
            px_jt=(p==t).float().mean().item()
            ex_jt=(p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
        print(f"    Joint TTT:      pixel={px_jt*100:.1f}% exact={ex_jt*100:.1f}%")
        results[comp]={'no_ttt':{'pixel':px_no,'exact':ex_no},'joint_ttt':{'pixel':px_jt,'exact':ex_jt}}
        _save(results)

    print(f"\n{'='*70}\nGRAND SUMMARY\n{'='*70}")
    for comp,r in results.items():
        print(f"  {comp}: no_ttt={r['no_ttt']['exact']*100:.0f}% -> joint={r['joint_ttt']['exact']*100:.0f}%")
    print("\nPhase 98 complete!")

if __name__=='__main__':main()
