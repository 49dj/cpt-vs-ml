import matplotlib.pyplot as plt

def plot_oof(observed,predicted,path):
    fig,ax=plt.subplots(figsize=(5,5))
    ax.scatter(observed,predicted,s=12)
    lo,hi=min(observed),max(observed)
    ax.plot([lo,hi],[lo,hi],color='black',linestyle='--')
    ax.set(xlabel='Observed Vs',ylabel='OOF predicted Vs')
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)
