"""
Phase 92: Liquid Mixture-of-Experts (L-MoE)

5 specialist L-NCAs (one per task). At test time:
  1. Forward demo through ALL specialists
  2. Route to the one with lowest loss
  3. Prompt-tune only that specialist

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
HC = 32; NCA_STEPS = 10; CTX_CH = 4; TTT_LR = 0.1; TTT_STEPS = 30
SPECIALIST_EPOCHS = 60; N_TRAIN = 1500; LR = 1e-3
TIME_BUDGET_MS = 500

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

def gen_color_invert(n,gs=8,seed=None):
    rng=np.random.RandomState(seed); ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(3,8)): r,c=rng.randint(0,gs,size=2); g[r,c]=rng.randint(1,4)
        res=g.copy(); res[g==1]=2; res[g==2]=1
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

def gen_move_right(n,gs=8,seed=None):
    rng=np.random.RandomState(seed); ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,6)): r,c=rng.randint(0,gs,size=2); g[r,c]=rng.randint(1,5)
        res=np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r,c]>0: res[r,(c+1)%gs]=g[r,c]
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

def gen_fill_border(n,gs=8,seed=None):
    rng=np.random.RandomState(seed); ins,tgs=[],[]
    for _ in range(n):
        g=np.zeros((gs,gs),dtype=np.int64)
        for _ in range(rng.randint(2,6)): r,c=rng.randint(0,gs,size=2); g[r,c]=rng.randint(1,5)
        res=g.copy(); res[0,:]=3; res[-1,:]=3; res[:,0]=3; res[:,-1]=3
        ins.append(to_onehot(g)); tgs.append(res)
    return torch.tensor(np.array(ins)),torch.tensor(np.array(tgs))

ALL_TASKS = [('gravity',gen_gravity),('expand',gen_expand),
             ('color_invert',gen_color_invert),('move_right',gen_move_right),
             ('fill_border',gen_fill_border)]


class PromptLNCA(nn.Module):
    def __init__(self, nc=10, hc=32, ctx_ch=4):
        super().__init__()
        self.hc=hc; self.ctx_proj=nn.Conv2d(ctx_ch,nc,1,bias=False)
        self.perceive=nn.Conv2d(nc+hc,hc*2,3,padding=1)
        self.update=nn.Sequential(nn.Conv2d(hc*2,hc,1),nn.ReLU(),nn.Conv2d(hc,hc,1))
        self.tau_gate=nn.Conv2d(nc+hc*2,hc,3,padding=1)
        self.b_tau=nn.Parameter(torch.ones(1,hc,1,1)*1.5)
        self.readout=nn.Conv2d(hc,nc,1)
        self.task_context=nn.Parameter(torch.zeros(1,ctx_ch,1,1))
    def forward(self, x, n_steps=NCA_STEPS):
        b,c,h,w=x.shape; ctx=self.ctx_proj(self.task_context.expand(b,-1,h,w)); x_aug=x+ctx
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


def train_specialist(gen_fn, task_name):
    """Train a single-task specialist."""
    x, y = gen_fn(N_TRAIN, GS, seed=SEED)
    model = PromptLNCA(hc=HC).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(SPECIALIST_EPOCHS):
        model.train(); perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BS):
            idx = perm[i:i+BS]; xb,yb = x[idx].to(DEVICE),y[idx].to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval()
    with torch.no_grad():
        p=model(x[:200].to(DEVICE)).argmax(1); acc=(p==y[:200].to(DEVICE)).float().mean()
    print(f"    {task_name}: train pixel={acc*100:.1f}%")
    return model.state_dict()


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase92_moe.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment':'Phase 92: L-MoE','timestamp':datetime.now().isoformat(),
                   'results':results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("="*70); print("Phase 92: Liquid Mixture-of-Experts (L-MoE)")
    print(f"  {len(ALL_TASKS)} specialists, routing by demo loss"); print("="*70)

    # 1. Train specialists
    print("\n  [1/3] Training specialists...")
    specialists = {}
    for tn, fn in ALL_TASKS:
        specialists[tn] = train_specialist(fn, tn)

    # 2. MoE Agent: route + prompt-tune
    print("\n  [2/3] Running MoE Agent on mixed test levels...")
    rng = np.random.RandomState(SEED+500)
    N_LEVELS = 25
    results = {'levels': []}
    total_solved = 0
    task_solved = {tn: [0,0] for tn,_ in ALL_TASKS}
    correct_routes = 0

    for level in range(N_LEVELS):
        task_idx = level % len(ALL_TASKS)
        true_task, gen_fn = ALL_TASKS[task_idx]
        gs = [8,10,12][level % 3]

        all_x, all_y = gen_fn(N_DEMO+1, gs, seed=SEED+600+level)
        demo_x, demo_y = all_x[:N_DEMO].to(DEVICE), all_y[:N_DEMO].to(DEVICE)
        test_x, test_y = all_x[N_DEMO:N_DEMO+1], all_y[N_DEMO:N_DEMO+1]

        t_start = time.perf_counter()

        # Route: find best specialist by demo loss
        best_loss, best_task = float('inf'), None
        for tn, state in specialists.items():
            model = PromptLNCA(hc=HC).to(DEVICE)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(model(demo_x), demo_y).item()
            if loss < best_loss:
                best_loss = loss; best_task = tn
            del model

        route_correct = (best_task == true_task)
        correct_routes += int(route_correct)

        # Prompt-tune the chosen specialist
        model = PromptLNCA(hc=HC).to(DEVICE)
        model.load_state_dict(specialists[best_task])
        model.freeze_backbone()
        model.task_context = nn.Parameter(torch.zeros(1,CTX_CH,1,1,device=DEVICE))
        opt = torch.optim.Adam([model.task_context], lr=TTT_LR)
        for _ in range(TTT_STEPS):
            if (time.perf_counter()-t_start)*1000 > TIME_BUDGET_MS*0.8: break
            opt.zero_grad(); F.cross_entropy(model(demo_x),demo_y).backward(); opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            p = model(test_x.to(DEVICE)).argmax(1); t = test_y.to(DEVICE)
            pixel = (p==t).float().mean().item()
            exact = (p.reshape(1,-1)==t.reshape(1,-1)).all(1).float().mean().item()
        total_ms = (time.perf_counter()-t_start)*1000
        solved = exact > 0.5
        total_solved += int(solved)
        task_solved[true_task][0] += int(solved)
        task_solved[true_task][1] += 1

        status = "SOLVED" if solved else "WRONG"
        route_mark = "OK" if route_correct else "MISS"
        print(f"    Lv{level:2d} ({true_task:>13s} {gs}x{gs}): {status:6s} "
              f"route={best_task}({route_mark}) pixel={pixel*100:.0f}% {total_ms:.0f}ms")

        results['levels'].append({
            'level':level,'task':true_task,'grid_size':gs,
            'routed_to':best_task,'route_correct':route_correct,
            'pixel_acc':pixel,'exact_match':exact,'total_ms':total_ms
        })
        del model; gc.collect()

    sr = total_solved / N_LEVELS
    route_acc = correct_routes / N_LEVELS
    avg_ms = np.mean([l['total_ms'] for l in results['levels']])
    results['summary'] = {
        'solve_rate':sr, 'solved':total_solved, 'total':N_LEVELS,
        'route_accuracy':route_acc, 'avg_latency_ms':avg_ms,
        'per_task':{k:{'solved':v[0],'total':v[1],'rate':v[0]/v[1] if v[1]>0 else 0}
                   for k,v in task_solved.items()}
    }
    _save(results)

    print(f"\n{'='*70}"); print("GRAND SUMMARY: L-MoE"); print(f"{'='*70}")
    print(f"  Solve Rate: {total_solved}/{N_LEVELS} = {sr*100:.0f}%")
    print(f"  Route Accuracy: {route_acc*100:.0f}%")
    print(f"  Avg Latency: {avg_ms:.0f}ms")
    for tn, st in results['summary']['per_task'].items():
        print(f"    {tn:15s}: {st['solved']}/{st['total']} = {st['rate']*100:.0f}%")

    _gen_fig(results); print("\nPhase 92 complete!")

def _gen_fig(results):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
        pt = results['summary']['per_task']; tasks = list(pt.keys())
        rates = [pt[t]['rate']*100 for t in tasks]
        colors = ['#EC4899' if r>50 else '#9CA3AF' for r in rates]
        bars = ax1.bar(range(len(tasks)),rates,color=colors,edgecolor='white')
        for b,v in zip(bars,rates): ax1.text(b.get_x()+b.get_width()/2,v+2,f'{v:.0f}%',ha='center',fontweight='bold')
        ax1.set_xticks(range(len(tasks))); ax1.set_xticklabels(tasks,rotation=20,fontsize=8)
        ax1.set_ylabel('Solve Rate (%)'); ax1.set_ylim(0,115); ax1.set_title('Per-Task (MoE)',fontweight='bold'); ax1.grid(axis='y',alpha=0.3)
        lats = [l['total_ms'] for l in results['levels']]
        ax2.bar(range(len(lats)),lats,color=['#10B981' if l<500 else '#EF4444' for l in lats],edgecolor='white',linewidth=0.3)
        ax2.axhline(y=500,color='red',linestyle='--',alpha=0.7)
        ax2.set_xlabel('Level'); ax2.set_ylabel('Latency (ms)'); ax2.set_title('Latency',fontweight='bold'); ax2.grid(axis='y',alpha=0.3)
        sr = results['summary']['solve_rate']; ra = results['summary']['route_accuracy']
        fig.suptitle(f'Phase 92: L-MoE Agent\nSolve: {sr*100:.0f}% | Route Acc: {ra*100:.0f}%',fontsize=13,fontweight='bold')
        plt.tight_layout(); os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR,"phase92_moe.png"),bbox_inches='tight',dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e: print(f"  Figure error: {e}")

if __name__ == '__main__': main()
