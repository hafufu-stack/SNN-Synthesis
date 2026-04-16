"""
Phase 100: v20 Ultimate Liquid AGI - Final Dry Run

Integrates ALL innovations:
  - L-MoE with 5 specialist experts
  - Attractor Regularization (stable at any T)
  - Auto-T Early Stopping
  - Prompt Tuning TTT (freeze backbone)
  - Temporal NBS (K=7)

40 levels, 5 tasks, 3 grid sizes, 500ms budget.

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
FIGURES_DIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"figures")
DEVICE="cpu";SEED=2026;NC=10;HC=32;GS=8;BS=32
EPOCHS=50;N_TRAIN=1000;LR=1e-3;N_DEMO=3
T_MIN=5;T_MAX=20;LAMBDA_ATT=0.1;AUTO_T_THRESH=1e-3;MAX_T=30
CTX_CH=4;TTT_STEPS=20;TTT_LR=0.1
NBS_K=7;NBS_SIGMAS=[0.0,0.01,0.03,0.05,0.1,0.15,0.2]
TIME_BUDGET=500;N_LEVELS=40

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

ALL_TASKS=[('gravity',_gravity),('expand',_expand),('color_invert',_color_invert),
           ('move_right',_move_right),('fill_border',_fill_border)]

def gen_task(name,n,gs=8,seed=None):
    rng=np.random.RandomState(seed)
    fns=dict(ALL_TASKS)
    ins,tgs=fns[name](n,gs,rng)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))


class V20Agent(nn.Module):
    """Ultimate L-NCA with Attractor Reg + Auto-T + Temporal NBS."""
    def __init__(self,nc=10,hc=32,ctx_ch=4):
        super().__init__()
        self.hc=hc;self.ctx_proj=nn.Conv2d(ctx_ch,nc,1,bias=False)
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
        self.sigma=0.0

    def forward(self,x,ctx,max_steps=MAX_T,threshold=AUTO_T_THRESH):
        b,c,h,w=x.shape
        ctx_signal=self.ctx_proj(ctx.expand(b,-1,h,w));x_a=x+ctx_signal
        state=torch.zeros(b,self.hc,h,w,device=x.device)
        for t in range(max_steps):
            combined=torch.cat([x_a,state],1);delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x_a,state,delta],1)
            tau_logit=self.tau_gate(tau_in)+self.b_tau
            if self.sigma>0:tau_logit=tau_logit+torch.randn_like(tau_logit)*self.sigma
            beta=torch.sigmoid(tau_logit).clamp(0.01,0.99)
            new_state=beta*state+(1-beta)*delta
            if threshold and t>=3:
                mse=((new_state-state)**2).mean().item()
                if mse<threshold:state=new_state;break
            state=new_state
        return self.readout(state)

    def forward_train(self,x,ctx,n_steps):
        b,c,h,w=x.shape
        ctx_signal=self.ctx_proj(ctx.expand(b,-1,h,w));x_a=x+ctx_signal
        state=torch.zeros(b,self.hc,h,w,device=x.device);prev=state
        for _ in range(n_steps):
            combined=torch.cat([x_a,state],1);delta=self.update(self.perceive(combined))
            tau_in=torch.cat([x_a,state,delta],1)
            beta=torch.sigmoid(self.tau_gate(tau_in)+self.b_tau).clamp(0.01,0.99)
            prev=state;state=beta*state+(1-beta)*delta
        return self.readout(state),state,prev


def train_specialist(task_name):
    x,y=gen_task(task_name,N_TRAIN,GS,seed=SEED)
    model=V20Agent().to(DEVICE)
    ctx=nn.Parameter(torch.zeros(1,CTX_CH,1,1))
    opt=torch.optim.Adam(list(model.parameters())+[ctx],lr=LR)
    for ep in range(EPOCHS):
        model.train();perm=torch.randperm(N_TRAIN)
        for i in range(0,N_TRAIN,BS):
            idx=perm[i:i+BS];xb,yb=x[idx].to(DEVICE),y[idx].to(DEVICE)
            T=random.randint(T_MIN,T_MAX);opt.zero_grad()
            out,st,prev=model.forward_train(xb,ctx,T)
            loss=F.cross_entropy(out,yb)+LAMBDA_ATT*((st-prev)**2).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+[ctx],1.0);opt.step()
    return model.state_dict(),ctx.data.clone()


def _save(r):
    os.makedirs(RESULTS_DIR,exist_ok=True)
    with open(os.path.join(RESULTS_DIR,"phase100_v20_agent.json"),'w',encoding='utf-8') as f:
        json.dump({'experiment':'Phase 100: v20 Ultimate Liquid AGI',
                   'timestamp':datetime.now().isoformat(),'results':r},f,indent=2,default=str)

def main():
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    print("="*70)
    print("  Phase 100: v20 Ultimate Liquid AGI")
    print(f"  {N_LEVELS} levels | {TIME_BUDGET}ms budget | MoE+AttractorReg+AutoT+NBS")
    print("="*70)

    # Train 5 specialists with attractor reg
    print("\n  [1/3] Training specialists with Attractor Regularization...")
    specialists={};ctxs={};task_names=[tn for tn,_ in ALL_TASKS]
    backbone_state=None
    for tn in task_names:
        state,ctx=train_specialist(tn)
        specialists[tn]=state;ctxs[tn]=ctx
        backbone_state=state
        print(f"    {tn}: done")

    # [2/3] Run v20 Agent
    print(f"\n  [2/3] Running v20 Agent ({N_LEVELS} levels)...")
    results={'levels':[]};total_solved=0
    task_solved={tn:[0,0] for tn in task_names}

    for level in range(N_LEVELS):
        task_idx=level%len(ALL_TASKS)
        true_task=task_names[task_idx]
        gs=[8,10,12][level%3]

        x,y=gen_task(true_task,N_DEMO+1,gs,seed=SEED+1000+level)
        dx,dy=x[:N_DEMO].to(DEVICE),y[:N_DEMO].to(DEVICE)
        tx,ty=x[N_DEMO:N_DEMO+1],y[N_DEMO:N_DEMO+1]

        t_start=time.perf_counter()

        # Route: find best specialist
        best_loss,best_tn=float('inf'),None
        for tn in task_names:
            m=V20Agent().to(DEVICE);m.load_state_dict(specialists[tn]);m.eval()
            with torch.no_grad():
                loss=F.cross_entropy(m(dx[:1],ctxs[tn]),dy[:1]).item()
            if loss<best_loss:best_loss=loss;best_tn=tn
            del m

        # TTT: Prompt tune the chosen context
        model=V20Agent().to(DEVICE);model.load_state_dict(specialists[best_tn])
        for p in model.parameters():p.requires_grad=False
        best_ctx=nn.Parameter(ctxs[best_tn].clone())
        opt=torch.optim.Adam([best_ctx],lr=TTT_LR)
        for _ in range(TTT_STEPS):
            if (time.perf_counter()-t_start)*1000>TIME_BUDGET*0.6:break
            opt.zero_grad()
            out=model.forward_train(dx,best_ctx,10)[0]
            F.cross_entropy(out,dy).backward();opt.step()

        # NBS inference with Auto-T
        model.eval();vote=torch.zeros(1,NC,gs,gs,device=DEVICE)
        with torch.no_grad():
            txd=tx.to(DEVICE)
            for sigma in NBS_SIGMAS[:NBS_K]:
                if (time.perf_counter()-t_start)*1000>TIME_BUDGET:break
                model.sigma=sigma
                vote+=F.softmax(model(txd,best_ctx,max_steps=MAX_T,threshold=AUTO_T_THRESH),dim=1)
        model.sigma=0.0

        pred=vote.argmax(dim=1);target=ty.to(DEVICE)
        total_ms=(time.perf_counter()-t_start)*1000
        px=(pred==target).float().mean().item()
        ex=(pred.reshape(1,-1)==target.reshape(1,-1)).all(1).float().mean().item()
        solved=ex>0.5;total_solved+=int(solved)
        task_solved[true_task][0]+=int(solved);task_solved[true_task][1]+=1

        status="SOLVED" if solved else ("TIMEOUT" if total_ms>TIME_BUDGET else "WRONG")
        route_ok="OK" if best_tn==true_task else "MISS"
        print(f"  Lv{level:2d} ({true_task:>13s} {gs}x{gs}): {status:7s} "
              f"route={best_tn}({route_ok}) px={px*100:.0f}% {total_ms:.0f}ms")
        results['levels'].append({'level':level,'task':true_task,'grid_size':gs,
            'routed_to':best_tn,'route_correct':best_tn==true_task,
            'pixel':px,'exact':ex,'ms':total_ms,'timeout':total_ms>TIME_BUDGET})
        del model;gc.collect()

    # Summary
    sr=total_solved/N_LEVELS
    avg_ms=np.mean([l['ms'] for l in results['levels']])
    timeouts=sum(1 for l in results['levels'] if l['timeout'])
    route_acc=sum(1 for l in results['levels'] if l['route_correct'])/N_LEVELS

    results['summary']={
        'solve_rate':sr,'solved':total_solved,'total':N_LEVELS,
        'route_accuracy':route_acc,'avg_latency_ms':avg_ms,
        'timeout_rate':timeouts/N_LEVELS,
        'per_task':{tn:{'solved':v[0],'total':v[1],'rate':v[0]/v[1] if v[1]>0 else 0}
                   for tn,v in task_solved.items()}
    }
    _save(results)

    print(f"\n{'='*70}")
    print(f"  PHASE 100 GRAND FINALE: v20 Ultimate Liquid AGI")
    print(f"{'='*70}")
    print(f"  Overall Solve Rate: {total_solved}/{N_LEVELS} = {sr*100:.0f}%")
    print(f"  Route Accuracy: {route_acc*100:.0f}%")
    print(f"  Avg Latency: {avg_ms:.0f}ms (budget: {TIME_BUDGET}ms)")
    print(f"  Timeouts: {timeouts}/{N_LEVELS} = {timeouts/N_LEVELS*100:.0f}%")
    print(f"\n  Per-task:")
    for tn,st in results['summary']['per_task'].items():
        print(f"    {tn:15s}: {st['solved']}/{st['total']} = {st['rate']*100:.0f}%")

    _gen_fig(results)
    print(f"\n{'='*70}")
    print(f"  THE END. Phase 100 complete!")
    print(f"{'='*70}")

def _gen_fig(results):
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(18,6))
        levels=results['levels']

        # Per-task
        ax=axes[0];pt=results['summary']['per_task'];tasks=list(pt.keys())
        rates=[pt[t]['rate']*100 for t in tasks]
        colors=['#EC4899' if r>50 else '#9CA3AF' for r in rates]
        bars=ax.bar(range(len(tasks)),rates,color=colors,edgecolor='white')
        for b,v in zip(bars,rates):ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.0f}%',ha='center',fontweight='bold')
        ax.set_xticks(range(len(tasks)));ax.set_xticklabels(tasks,rotation=20,fontsize=8)
        ax.set_ylabel('Solve Rate (%)');ax.set_ylim(0,115);ax.set_title('Per-Task',fontweight='bold')
        ax.grid(axis='y',alpha=0.3)

        # Per-size
        ax=axes[1]
        for gs in [8,10,12]:
            sub=[l for l in levels if l['grid_size']==gs]
            sr=sum(1 for l in sub if l['exact']>0.5)/len(sub)*100
            ax.bar(f'{gs}x{gs}',sr,color='#3B82F6',edgecolor='white')
            ax.text(ax.patches[-1].get_x()+ax.patches[-1].get_width()/2,sr+2,f'{sr:.0f}%',ha='center',fontweight='bold')
        ax.set_ylabel('Solve Rate (%)');ax.set_ylim(0,115);ax.set_title('Per-Size',fontweight='bold')
        ax.grid(axis='y',alpha=0.3)

        # Latency
        ax=axes[2];lats=[l['ms'] for l in levels]
        ax.bar(range(len(lats)),lats,color=['#10B981' if l<TIME_BUDGET else '#EF4444' for l in lats],edgecolor='white',linewidth=0.3)
        ax.axhline(y=TIME_BUDGET,color='red',linestyle='--',alpha=0.7)
        ax.set_xlabel('Level');ax.set_ylabel('Latency (ms)');ax.set_title('Latency',fontweight='bold')
        ax.grid(axis='y',alpha=0.3)

        sr=results['summary']['solve_rate']
        fig.suptitle(f'Phase 100: v20 Ultimate Liquid AGI\n'
                    f'Solve Rate: {sr*100:.0f}% | Route: {results["summary"]["route_accuracy"]*100:.0f}% | '
                    f'Avg: {results["summary"]["avg_latency_ms"]:.0f}ms',fontsize=14,fontweight='bold')
        plt.tight_layout();os.makedirs(FIGURES_DIR,exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR,"phase100_v20_agent.png"),bbox_inches='tight',dpi=200)
        plt.close();print(f"  Figure saved")
    except Exception as e:print(f"  Figure error: {e}")

if __name__=='__main__':main()
