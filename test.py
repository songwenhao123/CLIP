import clip
import torch
from evaluation import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device=device)
model.eval()
eval_datasets =["Aircraft",  'CIFAR100', 'DTD', 'OxfordPet']

evaluate(model, preprocess, eval_datasets)
