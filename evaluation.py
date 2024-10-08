import clip
import torch.nn as nn
import torch
from tqdm import tqdm
import datasets
from datasets.common import get_dataloader, maybe_dictionarize
import torch.nn.functional as F


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    # print('pred',pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model):
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    for classname in classnames:
        texts = [template(classname) for template in templates] 
        texts = clip.tokenize(texts).cuda()  
        class_embeddings = model.encode_text(texts)  
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  
        class_embedding = class_embeddings.mean(dim=0)  
        class_embedding /= class_embedding.norm() 
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights



@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()
        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights
       
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()
    # print('dataset',dataset.classnames)  # vocabulary list
    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )

    dataloader = get_dataloader(
        dataset, is_train=False, image_encoder=image_enc
    )
    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights)

    print(f"Top-1 accuracy: {top1:.2f}")
    # print(f"Top-5 accuracy: {top5:.2f}")


def evaluate(image_classifier, val_preprocess, eval_datasets):
    if eval_datasets is None:
        return
    for i, dataset_name in enumerate(eval_datasets):
        print("Evaluating on", dataset_name) 
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location='/home/wenhao/experiments/CLIP/data',
            batch_size=64,
            batch_size_eval=128,
        )
        eval_single_dataset(image_classifier, dataset)
