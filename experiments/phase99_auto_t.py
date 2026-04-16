"""
Phase 99: Auto-T Early Stopping

Stop NCA evolution when state converges: MSE(state_t, state_{t-1}) < threshold.
Combine with Attractor Regularization from Phase 95.

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
DEVICE="cpu";SEED=2026;NC=10;HC=32;GS=8;BS=32
EPOCHS=60;N_TRAIN=1000;N_TEST=200;LR=1e-3
T_MIN=5;T_MAX=20;LAMBDA_ATT=0.1;MAX_T=50
THRESHOLDS=[1e-2,1e-3,1e-4,1e-5]

def oh(g):
    h,w=g.shape;o=np.zeros((NC,h,w),dtype=np.float32)
    for c in range(NC):o[c]=(g==c).astype(np.float32)
    return o

def gen_expand(n,gs=8,seed=None):
    rng=np.random.RandomState(seed);ins,tgs=[],[]
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
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

def gen_gravity(n,gs=8,seed=None):
    rng=np.random.RandomState(seed);ins,tgs=[],[]
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
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))


class AutoTLNCA(nn.Module):
    def __init__(self,nc=10,hc=32):
        super().__init__()
        self.hc=hc
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)

    def forward(self,x,max_steps=50,threshold=None,return_steps=False):
        b,c,h,w=x.shape;state=torch.zeros(b,self.hc,h,w,device=x.device)
        actual_steps=0
        for t in range(max_steps):
            combined=torch.cat([x,state],1);delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            new_state=beta*state+(1-beta)*delta
            actual_steps=t+1
            if threshold is not None and t>=3:
                mse=((new_state-state)**2).mean().item()
                if mse<threshold:
                    state=new_state;break
            state=new_state
        out=self.readout(state)
        if return_steps:return out,actual_steps
        return out

    def forward_train(self,x,n_steps,return_states=False):
        b,c,h,w=x.shape;state=torch.zeros(b,self.hc,h,w,device=x.device);states=[]
        for _ in range(n_steps):
            combined=torch.cat([x,state],1);delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            state=beta*state+(1-beta)*delta
            if return_states:states.append(state)
        out=self.readout(state)
        if return_states:return out,states
        return out

def _save(r):
    os.makedirs(RESULTS_DIR,exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase99_auto_t.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 99','timestamp':datetime.now().isoformat(),'results':r},f,indent=2,default=str)

def main():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    print("="*70);print("Phase 99: Auto-T Early Stopping");print("="*70)

    TASKS={'gravity':gen_gravity,'expand':gen_expand}
    results={}

    for task_name,gen_fn in TASKS.items():
        print(f"\n  === {task_name} ===")
        x_tr,y_tr=gen_fn(N_TRAIN,GS,seed=SEED)
        x_te,y_te=gen_fn(N_TEST,GS,seed=SEED+1)

        # Train with attractor reg
        model=AutoTLNCA().to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        for ep in range(EPOCHS):
            model.train();perm=torch.randperm(N_TRAIN)
            for i in range(0,N_TRAIN,BS):
                idx=perm[i:i+BS];xb,yb=x_tr[idx].to(DEVICE),y_tr[idx].to(DEVICE)
                T=random.randint(T_MIN,T_MAX);opt.zero_grad()
                out,states=model.forward_train(xb,T,return_states=True)
                loss=F.cross_entropy(out,yb)
                if len(states)>=2:loss=loss+LAMBDA_ATT*((states[-1]-states[-2])**2).mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()

        # Test with different thresholds
        task_res=[]
        model.eval()
        for thresh in [None]+THRESHOLDS:
            total_steps=[]
            with torch.no_grad():
                all_preds=[];all_steps=[]
                for i in range(0,N_TEST,50):
                    xb=x_te[i:i+50].to(DEVICE)
                    p,s=model(xb,max_steps=MAX_T,threshold=thresh,return_steps=True)
                    all_preds.append(p.argmax(1));all_steps.append(s)
                preds=torch.cat(all_preds);t=y_te.to(DEVICE)
                px=(preds==t).float().mean().item()
                ex=(preds.reshape(N_TEST,-1)==t.reshape(N_TEST,-1)).all(1).float().mean().item()
                avg_steps=np.mean(all_steps)

            # Latency benchmark
            t0=time.perf_counter()
            with torch.no_grad():
                for _ in range(20):model(x_te[:1].to(DEVICE),max_steps=MAX_T,threshold=thresh)
            lat=(time.perf_counter()-t0)/20*1000

            label=f"thresh={thresh}" if thresh else "fixed T=50"
            entry={'threshold':thresh,'pixel':px,'exact':ex,'avg_steps':avg_steps,'latency_ms':lat}
            task_res.append(entry)
            print(f"    {label:20s}: px={px*100:.1f}% ex={ex*100:.1f}% avg_T={avg_steps:.1f} lat={lat:.1f}ms")

        results[task_name]=task_res
        _save(results)
        del model;gc.collect()

    print(f"\n{'='*70}\nGRAND SUMMARY\n{'='*70}")
    for tn,entries in results.items():
        print(f"  {tn}:")
        for e in entries:
            t=e['threshold']
            print(f"    thresh={t}: exact={e['exact']*100:.1f}% avg_T={e['avg_steps']:.1f} lat={e['latency_ms']:.1f}ms")
    print("\nPhase 99 complete!")

if __name__=='__main__':main()
