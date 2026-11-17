"""AI FATE Phase A: local MRI classification. No dataset downloads."""
import argparse, csv, hashlib, json, random, sys, platform
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def find_root(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f'Dataset folder does not exist: {path}. Extract the ZIP first.')
    candidates = [p for p in [path, *path.rglob('data'), *path.rglob('train')]
                  if p.is_dir()]
    candidates += [p.parent for p in path.rglob('train') if p.is_dir()]
    found = sorted({p for p in candidates if all((p / s).is_dir() for s in ['train','val','test'])})
    if len(found) != 1:
        raise ValueError('Supply the folder containing train/, val/, test/ (or its enclosing extracted dataset folder).')
    return found[0]

def scan(root):
    rows = []
    for split in ['train', 'val', 'test']:
        for label, name in enumerate(CLASSES):
            folder = root / split / name
            files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXT)
            if not files:
                raise ValueError(f'No images in {folder}')
            for p in files:
                with Image.open(p) as im:
                    im = im.convert('RGB'); im.load()
                    digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                rows.append({'path': str(p.relative_to(root)), 'split': split,
                             'class': name, 'label': label, 'pixel_sha256': digest})
    return rows

def audit(rows):
    # Preserve official splits; exclude exact duplicate groups with inconsistent labels.
    # For cross-split duplicates, retain the held-out split (test > val > train).
    priority = {'test': 0, 'val': 1, 'train': 2}
    groups = {}
    for r in rows:
        groups.setdefault(r['pixel_sha256'], []).append(r)
    clean, removed = [], []
    for group in groups.values():
        if len({r['label'] for r in group}) > 1:
            removed.extend([{**r, 'reason': 'conflicting_labels'} for r in group]); continue
        group.sort(key=lambda r: (priority[r['split']], r['path']))
        clean.append(group[0])
        removed.extend([{**r, 'reason': 'exact_pixel_duplicate'} for r in group[1:]])
    return sorted(clean, key=lambda r: r['path']), removed

class PadSquare:
    def __call__(self, im):
        w, h = im.size; side = max(w, h)
        return ImageOps.expand(im, ((side-w)//2, (side-h)//2, (side-w+1)//2, (side-h+1)//2), fill=0)

def transform(training=False):
    steps = [PadSquare(), transforms.Resize((224,224))]
    if training:
        steps += [transforms.RandomRotation(8), transforms.RandomAffine(0, translate=(0.03,0.03))]
    return transforms.Compose(steps + [transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])])

class MRIDataset(Dataset):
    def __init__(self, root, rows, training=False):
        self.root, self.rows, self.tf = root, rows, transform(training)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(self.root / r['path']) as im:
            x = self.tf(im.convert('RGB'))
        return x, r['label']

def build_model(pretrained):
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    return m

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, probs = [], []
    for x, y in loader:
        probs.append(model(x.to(device)).softmax(1).cpu().numpy()); ys.extend(y.tolist())
    return np.asarray(ys), np.concatenate(probs)

def checkpoint_load(path, device):
    return torch.load(path, map_location=device, weights_only=True)

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root = find_root(args.data)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output is not empty: {out}. Choose a new --output directory.')
    out.mkdir(parents=True, exist_ok=True)
    raw = scan(root); rows, removed = audit(raw)
    write_csv(out/'manifest_original.csv', raw)
    write_csv(out/'manifest_used.csv', rows)
    write_json(out/'excluded_images.json', removed)
    splits = {s: [r for r in rows if r['split']==s] for s in ['train','val','test']}
    for s, rr in splits.items():
        if set(r['label'] for r in rr) != set(range(4)):
            raise ValueError(f'Split {s} missing a class after audit.')
    if len(splits['test']) < 100: raise ValueError('Need at least 100 held-out test images.')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('Device:', device, '|', torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU (slower)')
    counts = {s: {c: sum(r['class']==c for r in rr) for c in CLASSES} for s,rr in splits.items()}
    print(json.dumps(counts, indent=2)); print('Excluded exact duplicates/conflicts:', len(removed))
    write_json(out/'run_config.json', {**vars(args), 'data':str(root), 'device':str(device),
        'classes':CLASSES,'counts':counts,'python':platform.python_version(),'torch':str(torch.__version__),
        'limitation':'Image-level evaluation. Patient independence and near-duplicate absence are NOT established.'})
    if args.audit_only:
        print('Audit complete:', out); return
    # Deterministic, stratified sample chosen before training; no cherry-picking.
    from sklearn.model_selection import train_test_split
    test_labels = [r['label'] for r in splits['test']]
    if len(test_labels)==100: selected=np.arange(100)
    else: _,selected=train_test_split(np.arange(len(test_labels)),test_size=100,
                                     stratify=test_labels,random_state=args.seed)
    selected=sorted(selected)
    write_csv(out/'appendix_selection.csv', [splits['test'][i] for i in selected])
    gen = torch.Generator().manual_seed(args.seed)
    loaders = {s: DataLoader(MRIDataset(root,rr,s=='train'), batch_size=args.batch_size,
               shuffle=s=='train', num_workers=0, pin_memory=device.type=='cuda', generator=gen)
               for s,rr in splits.items()}
    model=build_model(True).to(device)
    counts_train=np.bincount([r['label'] for r in splits['train']],minlength=4)
    weights=torch.tensor(len(splits['train'])/(4*counts_train),dtype=torch.float32,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    history=[]; best=-1.0
    for phase, epochs, lr in [('head',args.head_epochs,1e-3),('fine_tune',args.epochs,1e-4)]:
        if epochs == 0: continue
        if phase=='fine_tune' and (out/'best_model.pt').exists():
            model.load_state_dict(checkpoint_load(out/'best_model.pt',device)['state_dict'])
        for p in model.parameters(): p.requires_grad=False
        for p in model.classifier.parameters(): p.requires_grad=True
        if phase=='fine_tune':
            for block in model.features[-3:]:
                for p in block.parameters(): p.requires_grad=True
        optimizer=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=1e-4)
        phase_best=-1.; stale=0
        for epoch in range(1,epochs+1):
            model.train()
            # Frozen ImageNet BN statistics in both phases, appropriate for small batches.
            model.features.eval()
            total=0.; n=0
            for x,y in tqdm(loaders['train'],desc=f'{phase} {epoch}/{epochs}'):
                x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(x),y); loss.backward(); optimizer.step()
                total+=loss.item()*len(y); n+=len(y)
            y,p=evaluate(model,loaders['val'],device)
            f1=float(f1_score(y,p.argmax(1),average='macro'))
            history.append({'phase':phase,'epoch':epoch,'train_loss':total/n,
                            'val_accuracy':float(accuracy_score(y,p.argmax(1))),'val_macro_f1':f1})
            write_csv(out/'history.csv',history)
            print(history[-1])
            if f1>best:
                best=f1
                torch.save({'state_dict':model.state_dict(),'classes':CLASSES,
                            'preprocessing':'RGB; square black padding; 224 resize; ImageNet normalization',
                            'val_macro_f1':best},out/'best_model.pt')
            if f1>phase_best+1e-4: phase_best=f1; stale=0
            else: stale+=1
            if stale>=args.patience: break
    model.load_state_dict(checkpoint_load(out/'best_model.pt',device)['state_dict'])
    y,p=evaluate(model,loaders['test'],device); pred=p.argmax(1)
    metrics={'accuracy':float(accuracy_score(y,pred)),
             'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
             'macro_f1':float(f1_score(y,pred,average='macro')),
             'classification_report':classification_report(y,pred,labels=list(range(4)),target_names=CLASSES,output_dict=True,zero_division=0)}
    # Binary tumor presence uses summed tumor probability with fixed threshold 0.5.
    binary_y=(y!=CLASSES.index('notumor')).astype(int)
    tumor_p=1-p[:,CLASSES.index('notumor')]; binary_pred=(tumor_p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(binary_y,binary_pred,labels=[0,1]).ravel()
    metrics['tumor_detection']={'threshold':.5,'sensitivity':float(tp/(tp+fn)),
        'specificity':float(tn/(tn+fp)), 'roc_auc':float(roc_auc_score(binary_y,tumor_p)),
        'false_negatives':int(fn),'false_positives':int(fp)}
    majority=int(np.argmax(counts_train))
    metrics['majority_baseline_accuracy']=float(np.mean(y==majority))
    write_json(out/'metrics.json',metrics)
    predictions=[]
    for i,r in enumerate(splits['test']):
        predictions.append({'image':r['path'],'actual':CLASSES[y[i]],'predicted':CLASSES[pred[i]],
          'correct':bool(y[i]==pred[i]),'model_score':float(p[i,pred[i]]),
          'tumor_probability':float(tumor_p[i]),'tumor_predicted_at_0_5':bool(binary_pred[i]),
          **{f'p_{c}':float(p[i,j]) for j,c in enumerate(CLASSES)}})
    write_csv(out/'predictions_all.csv',predictions)
    write_csv(out/'predictions_100.csv',[predictions[i] for i in selected])
    cm=confusion_matrix(y,pred,labels=list(range(4)))
    fig,ax=plt.subplots(figsize=(7,6)); ax.imshow(cm,cmap='Blues')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,
           xlabel='Predicted',ylabel='Actual',title='Held-out test confusion matrix')
    for i in range(4):
        for j in range(4): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot([r['val_macro_f1'] for r in history],label='Validation macro F1')
    ax.plot([r['val_accuracy'] for r in history],label='Validation accuracy')
    ax.set(xlabel='Epoch (zero-based)',ylabel='Score'); ax.legend(); fig.tight_layout()
    fig.savefig(out/'learning_curve.png',dpi=160); plt.close(fig)
    (out/'evaluation_summary.txt').write_text(classification_report(y,pred,target_names=CLASSES,zero_division=0)
        +'\n'+json.dumps(metrics['tumor_detection'],indent=2)
        +'\nResearch/education only. Scores are not calibrated clinical probabilities.\n'
        +'Exact pixel duplicates were audited; near duplicates and patient overlap may remain.\n',encoding='utf-8')
    print('Complete:',out); print('Test accuracy:',metrics['accuracy'],'Macro F1:',metrics['macro_f1'])

def predict(args):
    device=torch.device('cpu'); saved=checkpoint_load(args.model,device)
    model=build_model(False); model.load_state_dict(saved['state_dict']); model.eval()
    with Image.open(args.image) as im: x=transform()(im.convert('RGB')).unsqueeze(0)
    with torch.no_grad(): p=model(x).softmax(1)[0].numpy()
    print(json.dumps({'predicted':CLASSES[int(p.argmax())],
        'class_scores':dict(zip(CLASSES,map(float,p))),
        'tumor_probability':float(1-p[2]),'tumor_at_threshold_0_5':bool(1-p[2]>=.5)},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train'); t.add_argument('--data',required=True)
    t.add_argument('--output',default='results'); t.add_argument('--epochs',type=int,default=15)
    t.add_argument('--head-epochs',type=int,default=3); t.add_argument('--batch-size',type=int,default=16)
    t.add_argument('--patience',type=int,default=4); t.add_argument('--seed',type=int,default=42)
    t.add_argument('--cpu',action='store_true'); t.add_argument('--audit-only',action='store_true')
    p=sub.add_parser('predict'); p.add_argument('--model',default='results/best_model.pt'); p.add_argument('--image',required=True)
    args=parser.parse_args()
    if args.command=='train':
        if args.epochs<0 or args.head_epochs<0 or args.epochs+args.head_epochs<1 or args.batch_size<1 or args.patience<1:
            parser.error('Epoch total, batch size, and patience must be positive.')
        train(args)
    else: predict(args)

if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError) as e:
        print(f'ERROR: {e}',file=sys.stderr); sys.exit(1)
