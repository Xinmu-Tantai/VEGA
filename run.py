import os
import sys
import re
import json
import torch
import random
import datetime
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from alisuretool.Tools import Tools
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from dataset.randaugment import RandomAugment
from torchvision.transforms import InterpolationMode
from utils import MetricLogger, SmoothedValue
from clip_model import CreateModel
from nltk import pos_tag, word_tokenize
import nltk
import pynvml
nltk.data.path.append('nltk_data')
from copy import deepcopy
import spacy
# from AVE import StructureAwareMultiGranularityAlignment
# from SAMGA import LSRFormerBidirectional
fg_sge_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FG-SGE')
if fg_sge_dir not in sys.path:
    sys.path.insert(0, fg_sge_dir)
from F_SAMGA import FrequencyGuidedStructureAwareMultiGranularityAlignment
from AVE_FGSGE_DWT4 import FGSGE_LSRFormerBidirectional



import warnings
warnings.filterwarnings('ignore')

def seed_worker(worker_id): 
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RETrainDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
                pass
            pass
        pass

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = self.pre_caption(ann['caption'], self.max_words)
        label = torch.tensor(ann['label'])
        return image, caption, self.img_ids[ann['image_id']], label

    @staticmethod
    def pre_caption(caption, max_words):
        caption = re.sub(r"([,.'!?\"()*#:;~])", '', caption.lower(),).replace(
            '-', ' ').replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(r"\s{2,}", ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ') 
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])
        if not len(caption):
            raise ValueError("pre_caption yields invalid text")
        return caption

    pass


class REEvalDataset(Dataset):

    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(RETrainDataset.pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
            pass
        pass

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

    pass


class CreateDataset(object):

    def __init__(self):
        g = torch.Generator()
        g.manual_seed(config.seed if hasattr(config, "seed") else 2024)
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                              (0.26862954, 0.26130258, 0.27577711))
        self.pretrain_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config.image_res, scale=(0.5, 1.0),
                                         interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(), self.normalize])
        self.test_transform = transforms.Compose([transforms.Resize((config.image_res, config.image_res),
                                                                    interpolation=InterpolationMode.BICUBIC),
                                                  transforms.ToTensor(), self.normalize])

        self.train_dataset = RETrainDataset(config.train_file, self.train_transform, config.image_root)
        self.test_dataset = REEvalDataset(config.test_file, self.test_transform, config.image_root) 
        self.val_dataset = REEvalDataset(config.val_file, self.test_transform, config.image_root)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size_train,
            num_workers=4, pin_memory=True, shuffle=True, drop_last=True,
            worker_init_fn=seed_worker, generator=g, persistent_workers=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config.batch_size_test,
            num_workers=4, pin_memory=True, shuffle=False, drop_last=False,
            worker_init_fn=seed_worker, generator=g, persistent_workers=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size_test,
            num_workers=4, pin_memory=True, shuffle=False, drop_last=False,
            worker_init_fn=seed_worker, generator=g, persistent_workers=True
        )
        pass

    pass


class RSITRBaseline(nn.Module):

    def __init__(self):
        super().__init__()
        create_model = CreateModel()
        self.model = create_model.create_model_and_transforms("ViT-B/32", pretrained=config.pretrain_path_open_clip)
        print(f"Model class: {self.model.__class__}")
        self.tokenize = create_model.tokenize 
        self.embed_dim_adapter = 512 
        self.alignment_adapter = FrequencyGuidedStructureAwareMultiGranularityAlignment(
            embed_dim=self.embed_dim_adapter,
            hidden_dim=self.embed_dim_adapter,
            num_heads=8,
            num_prototypes=64,
            weights=getattr(config, "align_weights", (1.0, 0.8, 0.4)),
        ) 
        self.bi_adapter = FGSGE_LSRFormerBidirectional(dim=self.embed_dim_adapter, heads=8) 
        self.bi_adapter.enable_glb2loc = getattr(config, "bi_enable_glb2loc", True)
        self.bi_adapter.enable_loc2glb = getattr(config, "bi_enable_loc2glb", True) 
        self.alpha_adapter = nn.Parameter(torch.tensor(5e-2)) 
        self.gamma_global_adapter = nn.Parameter(torch.tensor(5e-2)) 
        self.local_score_scale = nn.Parameter(torch.tensor(1.0)) 
        self.align_loss_weight_adapter = 0.02 
        self.use_upsample_7to8_adapter = True


    def extract_nouns(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        nouns = [word for word, tag in tagged if tag.startswith('NN')]
        return nouns if nouns else [tokens[0]]

    def get_vis_emb(self, image):
        features = self.model.encode_image(image, normalize=True)
        if isinstance(features, tuple):
            patch_feats, global_feat = features
            return patch_feats, global_feat
        else:
            return None, features


    def get_txt_emb(self, text_ids, return_word_feats=False):
        if return_word_feats:
            word_feats, sent_feat = self.model.encode_text(text_ids, normalize=True, return_word_feats=True)
            return word_feats, sent_feat
        else:
            sent_feat = self.model.encode_text(text_ids, normalize=True, return_word_feats=False)
            return None, sent_feat   
        
    @staticmethod
    def _tokens_to_map(x_tokens: torch.Tensor): 
        B, N, C = x_tokens.shape 
        H = W = int(N ** 0.5)
        if H * W != N:
            H2 = W2 = int((N - 1) ** 0.5)
            if H2 * W2 == (N - 1): 
                x_tokens = x_tokens[:, 1:, :]
                N = N - 1
                H = W = H2
            else:
                raise AssertionError(f"N={N} is not a perfect square (or 1+square); cannot reshape to HxW.")
        x_map = x_tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        return x_map, H, W

    @staticmethod
    def _map_to_tokens(x_map: torch.Tensor): 
        B, C, H, W = x_map.shape
        x_tokens = x_map.flatten(2).transpose(1, 2).contiguous()
        return x_tokens

    def apply_bi_to_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor: 
        patch_map, H, W = self._tokens_to_map(patch_tokens)   
 
        if self.use_upsample_7to8_adapter and (H == 7 and W == 7):
            patch_map_up = F.interpolate(patch_map, size=(8, 8), mode='bilinear', align_corners=False)
            out_map_up = self.bi_adapter(patch_map_up)   
            out_map = F.interpolate(out_map_up, size=(7, 7), mode='bilinear', align_corners=False)
        else:
            out_map = self.bi_adapter(patch_map)  

        return self._map_to_tokens(out_map)   


    def forward(self, image, text_ids, raw_texts, idx=None, label=None): 
        patch_feats, global_img_feat = self.get_vis_emb(image)
        word_feats, global_txt_feat = self.get_txt_emb(text_ids, return_word_feats=True)
 
        if patch_feats is not None:
            if getattr(config, "use_bi", True):
                patch_feats_bi = self.apply_bi_to_patch_tokens(patch_feats)
            else:
                patch_feats_bi = patch_feats 
        if patch_feats_bi is not None:
            pooled_avem = F.normalize(patch_feats_bi.detach().mean(dim=1), dim=-1)
            mix = torch.tanh(self.gamma_global_adapter)                   
            global_img_feat = F.normalize(global_img_feat + mix * pooled_avem, dim=-1) 
        else:
            patch_feats_bi = None
 
        if (patch_feats_bi is not None) and getattr(config, "use_alignment", True):
            align_loss, fused_feats, _ = self.alignment_adapter(
                patch_feats_bi, word_feats, return_details=True
            )  
            self._last_align_loss = align_loss.detach()
  
            gate = torch.tanh(self.alpha_adapter)  
            aligned_patch = patch_feats_bi + gate * fused_feats  
 
            lambda_local_align_cl = getattr(config, "lambda_local_align_cl", 0.1)
            loss_local_align_cl = self.get_local_aligned_contrastive_loss(
                aligned_patch, word_feats, temperature=getattr(config, "local_temp", 0.07)
            )

        else: 
            align_loss = torch.zeros([], device=global_img_feat.device)
            self._last_align_loss = align_loss.detach()
            loss_local_align_cl = torch.zeros([], device=global_img_feat.device)
 
        loss_contrastive = self.get_contrastive_loss(global_img_feat, global_txt_feat, idx)
        loss_triplet = self.get_triplet_loss(global_img_feat, global_txt_feat)
 
        lambda_local = getattr(config, "lambda_local_align", self.align_loss_weight_adapter)
        align_loss_pos = F.softplus(align_loss)
        loss_fine = lambda_local * align_loss_pos + loss_local_align_cl


        return loss_contrastive, loss_triplet, loss_fine

 
    
    def get_fine_grained_loss(self, patch_feats, noun_feats_list, temperature=0.07, pool_type='mean'):

        B, N, D = patch_feats.shape
        device = patch_feats.device
        H = W = int(N**0.5)
         
        contrast_losses = []
        mask_losses = []
        
        for b in range(B):
            patch_feat_b = patch_feats[b]   
            noun_feats_b = noun_feats_list[b]    
            similarity = torch.matmul(noun_feats_b, patch_feat_b.transpose(0, 1))   
            similarity_2d = similarity.view(len(noun_feats_b), H, W) 
             
            for noun_idx in range(len(noun_feats_b)):
                noun_sim = similarity[noun_idx]  
                k_pos = 5  
                topk_sim, topk_idx = torch.topk(noun_sim, k_pos) 
                pos_feats = patch_feat_b[topk_idx]    
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[topk_idx] = False
                neg_feats = patch_feat_b[mask]   
                l_pos = torch.einsum('d,kd->k', noun_feats_b[noun_idx], pos_feats)  
                l_neg = torch.einsum('d,nd->n', noun_feats_b[noun_idx], neg_feats)  
                 
                logits = torch.cat([l_pos, l_neg]) 
                labels = torch.zeros(N, device=device)
                labels[:k_pos] = 1.0
                
                contrast_loss = -torch.sum(F.log_softmax(logits / temperature, dim=0) * labels)
                contrast_losses.append(contrast_loss)
             
            if pool_type == 'mean':
                similarity_pooled = torch.mean(similarity_2d, dim=0)   
            elif pool_type == 'max':
                similarity_pooled = torch.max(similarity_2d, dim=0)[0]   
 
            k = 5  
            topk_similar = torch.topk(similarity_pooled.view(-1), k)[1]
            pred_mask = torch.zeros_like(similarity_pooled)
            pred_mask.view(-1).scatter_(-1, topk_similar, 1)
             
            mask_loss = F.binary_cross_entropy_with_logits(
                similarity_pooled / temperature,
                pred_mask.float()
            )
            mask_losses.append(mask_loss)
         
        contrast_loss = torch.mean(torch.stack(contrast_losses))
        mask_loss = torch.mean(torch.stack(mask_losses)) 
        lambda_contrast = 0.6
        lambda_mask = 0.4
        total_loss = lambda_contrast * contrast_loss + lambda_mask * mask_loss
        
        return total_loss

    def get_local_aligned_contrastive_loss(self, aligned_patch, word_feats, temperature=0.07): 
        img_local = aligned_patch.mean(dim=1)          
        txt_local = word_feats.mean(dim=1)             

        img_local = F.normalize(img_local, dim=-1)
        txt_local = F.normalize(txt_local, dim=-1)

        logits = img_local @ txt_local.t() / temperature   
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i2t + loss_t2i)


    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        logits = image_feat @ text_feat.t()
        if hasattr(self.model, "logit_scale"):
            logits = self.model.logit_scale.exp() * logits
        else: 
            logits = logits

        if idx is None:
            labels = torch.arange(image_feat.shape[0], device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)
            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2.0


    def get_triplet_loss(self, image_feat, text_feat, margin=0.1): 
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat  = F.normalize(text_feat, dim=-1)

        scores = image_feat @ text_feat.t()         
        diag = scores.diag().view(scores.size(0), 1)   
        cost_s = (margin + scores - diag).clamp(min=0) 
        cost_im = (margin + scores - diag.t()).clamp(min=0)
 
        mask = torch.eye(scores.size(0), device=scores.device).bool()
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)
 
        denom = scores.size(0) * (scores.size(0) - 1)
        loss = (cost_s.sum() + cost_im.sum()) / (2.0 * max(1, denom))
        return loss

    pass
  
class Runner(object):

    def __init__(self):
        self.device = torch.device(config.device)

        self.model = RSITRBaseline()
        self.tokenize = self.model.tokenize
        self.model = self.model.to(self.device)
        self.best_ckpt_path = None
        self.best_score = float("-inf")

        self.set_trainable(self.model)
        Tools.print(f"learnable parameter num = {self.count_trainable_parameters()}", txt_path=config.log_filename)

        self.create_dataset = CreateDataset()

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr,
                               weight_decay=config.weight_decay, eps=1e-8, betas=(0.9, 0.98))
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.epochs * len(self.create_dataset.train_loader))

        self.best_weights = None
        self.best_val_score = 0
        self.warm_up_epochs = 2  
 
    @staticmethod
    def set_trainable(model): 
        for name, module in model.named_modules():
            module.eval()
            for p in module.parameters(recurse=False):
                p.requires_grad = False
 
        for name, module in model.named_modules():
            if 'adapter' in name:
                module.train()
                for p in module.parameters():
                    p.requires_grad = True
 
        for name, p in model.named_parameters():
            if 'adapter' in name:
                p.requires_grad = True

        return model


    @staticmethod
    def load_checkpoint(model, checkpoint):
        if checkpoint != '-1':
            checkpoint_value = torch.load(checkpoint, map_location='cpu')
            state_dict = checkpoint_value['model'] if 'model' in checkpoint_value.keys() else checkpoint_value
            msg = model.load_state_dict(state_dict, strict=False)
            print("missing", msg.missing_keys)
            print("unexp", msg.unexpected_keys)
            pass
        pass

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _now_str(self):
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _save_best(self, epoch, test_result): 
        ckpt_dir = getattr(config, "ckpt_dir", "./checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        dataset_name = getattr(config, "dataset_name", "DATA")
        score = float(test_result.get("r_mean", -1e9))
 
        if score <= self.best_score:
            return None
 
        ts = self._now_str()
        filename = f"best_{dataset_name}_{ts}_epoch{epoch}_rmean{score:.2f}.pth"
        save_path = os.path.join(ckpt_dir, filename)

        payload = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            "test_result": test_result,
            "config": {k: v for k, v in config.__dict__.items() if isinstance(v, (int, float, str, bool))}
        }
        torch.save(payload, save_path)
 
        if self.best_ckpt_path is not None and os.path.isfile(self.best_ckpt_path):
            try:
                os.remove(self.best_ckpt_path)
            except Exception as e:
                Tools.print(f"[Checkpoint] WARNING: cannot remove old best: {self.best_ckpt_path}, err={e}",
                            txt_path=config.log_filename)
 
        self.best_score = score
        self.best_ckpt_path = save_path

        Tools.print(f"[Checkpoint] NEW BEST saved: {save_path}", txt_path=config.log_filename)
        return save_path


    def train(self):
        Tools.print("Start training ", txt_path=config.log_filename)
        best_result = self.test()
        self.best_score = float(best_result.get("r_mean", -1e9))
        self.best_ckpt_path = None

        for epoch in range(0, config.epochs):
            train_stats = self.train_one_epoch(self.create_dataset.train_loader, epoch)
            test_result = self.test() 
            Tools.print(
                json.dumps({
                    "epoch": epoch,
                    "test": test_result
                }),
                txt_path=config.log_filename
            )
 
            saved = self._save_best(epoch=epoch, test_result=test_result)
            if saved is not None:
                best_result = test_result

        Tools.print(f"[Best] best_result={best_result}, best_ckpt={self.best_ckpt_path}",
                    txt_path=config.log_filename)
        return best_result



    def train_one_epoch(self, data_loader, epoch):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_contrastive', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_triplet', SmoothedValue(window_size=1, fmt='{value:.4f}')) 
        metric_logger.add_meter('loss_fine', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('align_loss_raw', SmoothedValue(window_size=1, fmt='{value:.4f}')) 

        header = 'Train Epoch: [{}]'.format(epoch) 

        self.model.eval()
        for name, module in self.model.named_modules():
            if 'adapter' in name:
                module.train()

        metric_logger.update(lr=0.0, loss_contrastive=0.0, loss_triplet=0.0, loss_fine=0.0, align_loss_raw=0.0, align_loss_ref=0.0)
        
        for i, (image, text, idx, label) in enumerate(metric_logger.log_every(data_loader, 50, header)):
            image = image.to(self.device, non_blocking=True)
            idx = idx.to(self.device, non_blocking=True)
            text_input = self.tokenize(text).to(self.device)

            loss_contrastive, loss_triplet, loss_fine = self.model(
                image, text_input, text, idx=idx, label=label
            )
            loss = loss_contrastive + loss_triplet + loss_fine

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            metric_logger.update(loss_contrastive=float(loss_contrastive.item()))
            metric_logger.update(loss_triplet=float(loss_triplet.item()))
            metric_logger.update(loss_fine=float(loss_fine.item()))
            metric_logger.update(lr=float(self.optimizer.param_groups[0]["lr"]))
 
            if hasattr(self.model, "_last_align_loss"):
                metric_logger.update(align_loss_raw=float(self.model._last_align_loss.item()))
            if hasattr(self.model, "_last_align_loss_refined"):
                metric_logger.update(align_loss_ref=float(self.model._last_align_loss_refined.item()))

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def test(self):
        score_test_i2t, score_test_t2i = self.evaluation(self.create_dataset.test_loader)
        test_result = self.itm_eval(score_test_i2t, score_test_t2i, self.create_dataset.test_dataset.txt2img,
                                    self.create_dataset.test_dataset.img2txt) 
        print(f"[Eval] test_result={test_result}")
        return test_result
    
    @torch.no_grad() 
    def batched_aggregate_similarity(self, patch_noun_sim_batch, method='mean', mask=None, temperature=0.07, top_k=5):
        B, num_X, num_Y = patch_noun_sim_batch.shape
        
        if num_X == 0 or num_Y == 0:
            return torch.zeros(B, device=patch_noun_sim_batch.device, dtype=patch_noun_sim_batch.dtype)

        processed_sim_batch = patch_noun_sim_batch 
        
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(-1, num_X, -1)
            processed_sim_batch = processed_sim_batch.masked_fill(~expanded_mask, -1e9)
        patch_noun_sim_flat = processed_sim_batch.reshape(B, -1)

        
        flat_mask = None
        if mask is not None:
            flat_mask = expanded_mask.reshape(B, -1)
        if method == 'max':
            return torch.max(patch_noun_sim_flat, dim=1)[0]
        
        elif method == 'mean':
            if mask is not None:
                masked_sims = patch_noun_sim_batch.masked_fill(~expanded_mask, 0)
                sum_of_sims = masked_sims.sum(dim=(-1,-2)) 
                num_valid_elements_per_matrix = expanded_mask.sum(dim=(-1,-2))
                num_valid_elements_per_matrix = num_valid_elements_per_matrix.clamp(min=1)
                return sum_of_sims / num_valid_elements_per_matrix
            else:
                return torch.mean(patch_noun_sim_flat, dim=1)
        

        return torch.max(patch_noun_sim_flat, dim=1)[0]

    @torch.no_grad()
    def alignment_local_score(self, img_patch: torch.Tensor, txt_word: torch.Tensor): 
        align_loss, fused_feats, _ = self.model.alignment_adapter(img_patch, txt_word, return_details=True) 
        gate = torch.tanh(self.model.alpha_adapter)  
        aligned_patch = img_patch + gate * fused_feats   
        aligned_patch = F.normalize(aligned_patch, dim=-1)
        txt_word = F.normalize(txt_word, dim=-1)
 
        sim = torch.einsum("bnd,bmd->bnm", aligned_patch, txt_word) 
        scale = getattr(self.model, "local_score_scale", None)
        if scale is not None:
            sim = sim * scale.clamp(0.1, 10.0) 
        tau = getattr(config, "local_agg_tau", 0.03)
        tau = float(tau) if tau is not None else 0.03
        tau = max(1e-4, tau)
        token_score = torch.logsumexp(sim / tau, dim=1) * tau  # [B, Nt]
        score = token_score.mean(dim=1)  # [B]
        return score


    @torch.no_grad()
    def evaluation(self, data_loader):
        self.model.eval()
        device = next(self.model.parameters()).device 
        alpha = getattr(config, "eval_alpha", 1) 
        topk = getattr(config, "eval_topk_align", 128)
 
        image_patch_embeds_list = []
        image_global_embeds_list = []
        for image, _ in data_loader:
            patch_embed, global_embed = self.model.get_vis_emb(image.to(device))
            if patch_embed is not None:
                if getattr(config, "use_bi", True):
                    patch_embed = self.model.apply_bi_to_patch_tokens(patch_embed) 
                    pooled_avem = F.normalize(patch_embed.mean(dim=1), dim=-1)   
                    mix = torch.tanh(self.model.gamma_global_adapter)           
                    global_embed = F.normalize(global_embed + mix * pooled_avem, dim=-1)

                image_patch_embeds_list.append(patch_embed)

            image_global_embeds_list.append(global_embed)  
        image_patch_embeds = torch.cat(image_patch_embeds_list, dim=0) if image_patch_embeds_list else None
        image_global_embeds = torch.cat(image_global_embeds_list, dim=0)
 
        text_word_embeds = []
        text_global_embeds = []
        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = config.batch_size_test_text

        for i in range(0, num_text, text_bs):
            text_batch = texts[i: min(num_text, i + text_bs)]
            text_input = self.tokenize(text_batch).to(device)
            word_embed, global_embed = self.model.get_txt_emb(text_input, return_word_feats=True)
            text_word_embeds.append(word_embed)
            text_global_embeds.append(global_embed)

        text_word_embeds = torch.cat(text_word_embeds, dim=0)     
        text_global_embeds = torch.cat(text_global_embeds, dim=0)   

        N_img = image_global_embeds.shape[0]
        N_txt = text_global_embeds.shape[0]
 
        sims_matrix_global = image_global_embeds @ text_global_embeds.t()  
        force_rerank = getattr(config, "force_rerank", True) and getattr(config, "use_alignment", True)
        if ((not getattr(config, "use_eval_rerank", True)) and (not force_rerank)) or (image_patch_embeds is None):
            return sims_matrix_global.cpu().numpy(), sims_matrix_global.t().cpu().numpy()

 
        if image_patch_embeds is None:
            final_i2t = sims_matrix_global
            final_t2i = sims_matrix_global
            return final_i2t.cpu().numpy(), final_t2i.t().cpu().numpy()
 
        calib = getattr(config, "eval_local_calibrate", True)
        calib_mode = getattr(config, "eval_local_calib_mode", "z_to_global")
        protect_topk = getattr(config, "eval_rerank_protect_topk", True) 

        def _calibrate_local(scores_local: torch.Tensor, scores_global_cand: torch.Tensor) -> torch.Tensor:
            """Calibrate local scores to be on a comparable scale to candidate global scores."""
            if (not calib) or (scores_local.numel() <= 1):
                return scores_local
            eps = 1e-6
            if calib_mode == "zscore":
                mu = scores_local.mean()
                sd = scores_local.std(unbiased=False).clamp_min(eps)
                return (scores_local - mu) / sd
            if calib_mode == "minmax":
                mn = scores_local.min()
                mx = scores_local.max()
                denom = (mx - mn).clamp_min(eps)
                return (scores_local - mn) / denom 
            mu_l = scores_local.mean()
            sd_l = scores_local.std(unbiased=False).clamp_min(eps)
            z = (scores_local - mu_l) / sd_l
            mu_g = scores_global_cand.mean()
            sd_g = scores_global_cand.std(unbiased=False).clamp_min(eps)
            return z * sd_g + mu_g

        def _write_back_protected(final_scores_row: torch.Tensor,
                                  cand: torch.Tensor,
                                  global_cand: torch.Tensor,
                                  mix_cand: torch.Tensor): 
            order = torch.argsort(mix_cand, descending=True)                 
            cand_sorted_by_mix = cand[order]                                 
            global_sorted = torch.sort(global_cand, descending=True).values  
            final_scores_row[cand_sorted_by_mix] = global_sorted

        final_sims_for_i2t = sims_matrix_global.clone()
        final_sims_for_t2i = sims_matrix_global.clone()
 
        K_i2t = min(topk, N_txt)
        for i in range(N_img):
            cand = torch.topk(sims_matrix_global[i], k=K_i2t, largest=True).indices  
            img_patch = image_patch_embeds[i].unsqueeze(0).expand(K_i2t, -1, -1)    
            txt_word = text_word_embeds[cand]                                       
            scores_local = self.alignment_local_score(img_patch, txt_word)          

            global_cand = sims_matrix_global[i, cand].detach()
            scores_local = _calibrate_local(scores_local, global_cand)
            mix_cand = alpha * global_cand + (1.0 - alpha) * scores_local
 
            if protect_topk:
                _write_back_protected(final_sims_for_i2t[i], cand, global_cand, mix_cand)
            else:
                final_sims_for_i2t[i, cand] = mix_cand
 
        K_t2i = min(topk, N_img)
        for t in range(N_txt):
            cand = torch.topk(sims_matrix_global[:, t], k=K_t2i, largest=True).indices  
            img_patch = image_patch_embeds[cand]                                         
            txt_word = text_word_embeds[t].unsqueeze(0).expand(K_t2i, -1, -1)            
            scores_local = self.alignment_local_score(img_patch, txt_word)               

            global_cand = sims_matrix_global[cand, t].detach()
            scores_local = _calibrate_local(scores_local, global_cand)
            mix_cand = alpha * global_cand + (1.0 - alpha) * scores_local
 
            if protect_topk:
                _write_back_protected(final_sims_for_t2i[:, t], cand, global_cand, mix_cand)
            else:
                final_sims_for_t2i[cand, t] = mix_cand

        return final_sims_for_i2t.cpu().numpy(), final_sims_for_t2i.t().cpu().numpy()

    
    def extract_nouns(self, text):

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        nouns = [word for word, tag in tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    

        if len(nouns) > 1:

            from collections import Counter
            noun_counts = Counter(nouns)

            threshold = max(noun_counts.values()) * 0.5
            nouns = [noun for noun, count in noun_counts.items() if count >= threshold]
    
        return nouns if nouns else [tokens[0]]
        
    @torch.no_grad()
    def itm_eval(self, scores_i2t, scores_t2i, txt2img, img2txt): 
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            if torch.is_tensor(score):
                score = score.detach().float().cpu().numpy()
            inds = np.argsort(score)[::-1]
 
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            pass
 
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
 
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            if torch.is_tensor(score):
                score = score.detach().float().cpu().numpy()
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]
            pass
 
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        eval_result = {'txt_r1': round(tr1, 2),
                       'txt_r5': round(tr5, 2),
                       'txt_r10': round(tr10, 2),
                       'img_r1': round(ir1, 2),
                       'img_r5': round(ir5, 2),
                       'img_r10': round(ir10, 2),
                       'r_mean': round(r_mean, 2)}
        return eval_result

    pass


class ConfigCommon(object):

    def __init__(self):
        self.clean_gpu()
        self.gpu_id=0
        torch.cuda.set_device(self.gpu_id) 
        self.seed = 2024
        self.setup_seed(self.seed)
        self.eval_alpha = 0.98
        self.lambda_local_align = 0.05 
        self.eval_topk_align = 256          
        self.device = "cuda"
        self.epochs = 7
        self.lr = 0.001
        self.weight_decay = 0.01 
        self.image_res = 224   
        self.patch_size = 32    
        self.pretrain_path_open_clip = "open_clip_pytorch_model.bin"
        self.batch_size_train = 256
        self.batch_size_test = 128
        self.batch_size_test_text = 128
        pass

    @staticmethod
    def setup_seed(seed): 
        os.environ["PYTHONHASHSEED"] = str(seed) 
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
        random.seed(seed)
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        torch.use_deterministic_algorithms(True)


    @staticmethod
    def get_gpu_id(): 
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_id, free = 0, 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            now_free = (info.free // 1048576) / 1024   
            if now_free > free:
                free = now_free
                gpu_id = i
            pass
        pynvml.nvmlShutdown()
        return gpu_id

    @staticmethod
    def clean_gpu(o=None):
        if o is not None:
            del o
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        pass

    pass

class Config_RSITMD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()
        self.output_dir = Tools.new_dir("./outputs/test_RSITMD_ViT")
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))
        self.dataset_name = "RSITMD"   # RSITMD/UCM 类似
        self.ckpt_dir = Tools.new_dir(os.path.join(self.output_dir, "checkpoints"))

        self.model = 'vit'
        self.lr = 0.001
 
        self.image_root = './dataset/RSITMD'
        self.train_file = ['data/finetune/rsitmd_train.json']   
        self.val_file = 'data/finetune/rsitmd_val.json'  
        self.test_file = 'data/finetune/rsitmd_test.json'   
        pass

    pass


class Config_RSICD_ViT(ConfigCommon):

    def __init__(self):
        super().__init__()

        self.output_dir = Tools.new_dir("./outputs/test_RSICD_ViT")
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))

        self.dataset_name = "RSICD"   # RSITMD/UCM 类似
        self.ckpt_dir = Tools.new_dir(os.path.join(self.output_dir, "checkpoints")) 

        self.model = 'vit'
        self.lr = 0.001
 
        self.image_root = './dataset/RSICD'
        self.train_file = ['data/finetune/rsicd_train.json']   
        self.val_file = 'data/finetune/rsicd_val.json'   
        self.test_file = 'data/finetune/rsicd_test.json'   
        pass

    pass

class Config_UCM_Caption_ViT(ConfigCommon):

    def __init__(self):
        super().__init__() 

        self.output_dir = Tools.new_dir("./outputs/test_UCM_ViT") 
        self.log_filename = os.path.join(self.output_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-log.txt"))
        self.dataset_name = "UCM"   # RSITMD/UCM 类似
        self.ckpt_dir = Tools.new_dir(os.path.join(self.output_dir, "checkpoints"))
        self.model = 'vit' 
        self.lr = 0.001  
        self.image_root = './dataset/UCM'
        converted_json_dir = './data/finetune/' 
        self.train_file = [os.path.join(converted_json_dir, 'ucm_caption_train_converted.json')]
        self.val_file = os.path.join(converted_json_dir, 'ucm_caption_val_converted.json')
        self.test_file = os.path.join(converted_json_dir, 'ucm_caption_test_converted.json')
    
        pass 

    pass 


if __name__ == '__main__':

    result_list = [] 
    for ConfigCLS in [Config_RSITMD_ViT]: 
        config = ConfigCLS()
        runner = Runner()
        best_result = runner.train()
        result_list.append(best_result)
        pass

    for result_one in result_list:
        Tools.print(result_one)
        pass
    pass
