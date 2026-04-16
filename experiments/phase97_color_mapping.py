"""
Phase 97: Color-Frequency Mapping

Remap colors by frequency before L-NCA input.
Most frequent color = ID 0, next = ID 1, etc.

Author: Hiroto Funasaki
"""
import os,json,time,gc,random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings;warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"results")
DEVICE="cpu";SEED=2026;NC=10;HC=32;NCA_STEPS=10;GS=8;BS=32
EPOCHS=50;N_TRAIN=1000;N_TEST=200;LR=1e-3

def oh(g):
    h,w=g.shape;o=np.zeros((NC,h,w),dtype=np.float32)
    for c in range(NC):o[c]=(g==c).astype(np.float32)
    return o

def freq_remap(grid):
    """Remap colors by frequency: most frequent -> 0, next -> 1, etc."""
    flat=grid.flatten()
    unique,counts=np.unique(flat,return_counts=True)
    # Sort by count descending, then by value for ties
    order=np.argsort(-counts)
    color_map={}
    for new_id,idx in enumerate(order):
        color_map[unique[idx]]=new_id
    remapped=np.vectorize(color_map.get)(grid)
    return remapped.astype(np.int64),color_map

def gen_gravity_color_shifted(n,gs=8,seed=None,shift=0):
    """Gravity task but with shifted colors (e.g., color 3 falls instead of 1)."""
    rng=np.random.RandomState(seed);ins,tgs=[],[]
    fall_color=(1+shift)%NC;fix_color=(2+shift)%NC
    if fall_color==0:fall_color=1
    if fix_color==0:fix_color=2
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):g[rng.randint(0,gs),rng.randint(0,gs)]=fall_color
        for _ in range(rng.randint(1,3)):
            r,c=rng.randint(0,gs),rng.randint(0,gs)
            if g[r,c]==0:g[r,c]=fix_color
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]==fix_color:res[r,c]=fix_color
        for c in range(gs):
            cnt=sum(1 for r in range(gs) if g[r,c]==fall_color);row,pl=gs-1,0
            while pl<cnt and row>=0:
                if res[row,c]==0:res[row,c]=fall_color;pl+=1
                row-=1
        ins.append(oh(g));tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

def gen_with_remap(n,gs=8,seed=None,shift=0):
    """Generate gravity with shifted colors, then remap both input and target."""
    rng=np.random.RandomState(seed);ins,tgs=[],[]
    fall_color=(1+shift)%NC;fix_color=(2+shift)%NC
    if fall_color==0:fall_color=1
    if fix_color==0:fix_color=2
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,5)):g[rng.randint(0,gs),rng.randint(0,gs)]=fall_color
        for _ in range(rng.randint(1,3)):
            r,c=rng.randint(0,gs),rng.randint(0,gs)
            if g[r,c]==0:g[r,c]=fix_color
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]==fix_color:res[r,c]=fix_color
        for c in range(gs):
            cnt=sum(1 for r in range(gs) if g[r,c]==fall_color);row,pl=gs-1,0
            while pl<cnt and row>=0:
                if res[row,c]==0:res[row,c]=fall_color;pl+=1
                row-=1
        g_remap,cmap=freq_remap(g)
        res_remap=np.vectorize(cmap.get)(res).astype(np.int64)
        ins.append(oh(g_remap));tgs.append(res_remap)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))


class LNCA(nn.Module):
    def __init__(self,nc=10,hc=32):
        super().__init__()
        self.hc=hc
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
    def forward(self,x,n_steps=NCA_STEPS):
        b,c,h,w=x.shape;state=torch.zeros(b,self.hc,h,w,device=x.device)
        for _ in range(n_steps):
            combined=torch.cat([x,state],1);delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
        return self.readout(state)


def train_and_eval(x_train,y_train,x_test,y_test,label):
    model=LNCA().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=LR)
    n=x_train.size(0)
    for ep in range(EPOCHS):
        model.train();perm=torch.randperm(n)
        for i in range(0,n,BS):
            idx=perm[i:i+BS];opt.zero_grad()
            F.cross_entropy(model(x_train[idx].to(DEVICE)),y_train[idx].to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    model.eval()
    with torch.no_grad():
        p=model(x_test.to(DEVICE)).argmax(1);t=y_test.to(DEVICE)
        px=(p==t).float().mean().item()
        ex=(p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
    print(f"    {label}: pixel={px*100:.1f}% exact={ex*100:.1f}%")
    del model;gc.collect()
    return {'pixel':px,'exact':ex}

def _save(r):
    os.makedirs(RESULTS_DIR,exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase97_color_mapping.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 97','timestamp':datetime.now().isoformat(),'results':r},f,indent=2,default=str)

def main():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    print("="*70);print("Phase 97: Color-Frequency Mapping");print("="*70)
    results={}

    # Train on standard colors (shift=0)
    print("\n  [1] Train on standard gravity (color 1 falls)")
    x_tr,y_tr=gen_gravity_color_shifted(N_TRAIN,GS,seed=SEED,shift=0)
    x_te,y_te=gen_gravity_color_shifted(N_TEST,GS,seed=SEED+1,shift=0)
    results['baseline_same']=train_and_eval(x_tr,y_tr,x_te,y_te,"Same colors")

    # Test on shifted colors (shift=3: color 4 falls)
    print("\n  [2] Test baseline on shifted colors (color 4 falls)")
    x_te_s,y_te_s=gen_gravity_color_shifted(N_TEST,GS,seed=SEED+1,shift=3)
    model=LNCA().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=LR)
    for ep in range(EPOCHS):
        model.train();perm=torch.randperm(N_TRAIN)
        for i in range(0,N_TRAIN,BS):
            idx=perm[i:i+BS];opt.zero_grad()
            F.cross_entropy(model(x_tr[idx].to(DEVICE)),y_tr[idx].to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    model.eval()
    with torch.no_grad():
        p=model(x_te_s.to(DEVICE)).argmax(1);t=y_te_s.to(DEVICE)
        px=(p==t).float().mean().item()
        ex=(p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
    results['baseline_shifted']={'pixel':px,'exact':ex}
    print(f"    Shifted colors (no remap): pixel={px*100:.1f}% exact={ex*100:.1f}%")
    del model;gc.collect()

    # Train with freq-remap
    print("\n  [3] Train with freq-remap, test on shifted colors (remapped)")
    x_tr_r,y_tr_r=gen_with_remap(N_TRAIN,GS,seed=SEED,shift=0)
    x_te_r,y_te_r=gen_with_remap(N_TEST,GS,seed=SEED+1,shift=3)
    results['remap_shifted']=train_and_eval(x_tr_r,y_tr_r,x_te_r,y_te_r,"Freq-remapped shifted")

    # Cross-shift test
    print("\n  [4] Cross-shift generalization (train shift=0, test shift=1..5)")
    cross_results=[]
    for s in range(1,6):
        x_te_c,y_te_c=gen_with_remap(N_TEST,GS,seed=SEED+1,shift=s)
        model=LNCA().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=LR)
        for ep in range(EPOCHS):
            model.train();perm=torch.randperm(N_TRAIN)
            for i in range(0,N_TRAIN,BS):
                idx=perm[i:i+BS];opt.zero_grad()
                F.cross_entropy(model(x_tr_r[idx].to(DEVICE)),y_tr_r[idx].to(DEVICE)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        model.eval()
        with torch.no_grad():
            p=model(x_te_c.to(DEVICE)).argmax(1);t=y_te_c.to(DEVICE)
            px=(p==t).float().mean().item()
            ex=(p.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
        cross_results.append({'shift':s,'pixel':px,'exact':ex})
        print(f"    shift={s}: pixel={px*100:.1f}% exact={ex*100:.1f}%")
        del model;gc.collect()
    results['cross_shift']=cross_results
    _save(results)

    print(f"\n{'='*70}\nGRAND SUMMARY\n{'='*70}")
    print(f"  Same colors:       exact={results['baseline_same']['exact']*100:.1f}%")
    print(f"  Shifted (no remap): exact={results['baseline_shifted']['exact']*100:.1f}%")
    print(f"  Shifted (remapped): exact={results['remap_shifted']['exact']*100:.1f}%")
    print("\nPhase 97 complete!")

if __name__=='__main__':main()
