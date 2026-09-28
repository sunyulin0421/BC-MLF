import os
import re
from contextlib import nullcontext
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau


from ...utils import MetricsTop, dict_to_str
from ...utils.schedulers import get_linear_schedule_with_warmup, get_scheduler

logger = logging.getLogger('MMSA')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# __all__ = ['MMSeq2Seq']


class MSALM():
    def __init__(self, args):
        self.args = args
        # bf16 - training
        self.use_bf16 = self.args.get("use_bf16", False)
        if self.use_bf16:
            self.scaler = amp.GradScaler()

        # lm-flavor
        self.lm_flavor = self.args["mmgpt"].get("type", "llama")

        # modified loss fot bn version
        self.modded_loss = self.args.get("modded_loss", False)
        self.use_cmc_loss = self.args.get("av_distil", False)
        self.use_clm_loss = self.args.get("use_clm", False)
        self.n_bn_fusion = self.args.get("n_bn_fusion", -1)
        self.use_fusion_contrast = self.args.get("use_fusion_contrast", False)
        self.lambda_fusion_contrast = self.args.get("lambda_fusion_contrast", 0.0)
        self.contrast_temperature = self.args.get("contrast_temperature", 0.2)
        self.contrast_label_sigma = self.args.get("contrast_label_sigma", 1.0)
        self.dump_baseline_diagnostics = self.args.get("dump_baseline_diagnostics", False)
        self.dump_branch_occlusion_diagnostics = self.args.get(
            "dump_branch_occlusion_diagnostics", False
        )
        self.dump_diagnostics_mode = str(self.args.get("dump_diagnostics_mode", "TEST")).upper()
        self.dump_diagnostics_dir = self.args.get("dump_diagnostics_dir", "diagnostics")
        self.dump_diagnostics_prefix = self.args.get("dump_diagnostics_prefix", "")
        self.dump_bn_calibration_beta = float(self.args.get("dump_bn_calibration_beta", 0.1))

        self.criterion = nn.L1Loss(reduction='none') if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        # extra losses
        if self.modded_loss:
            self.crit_text = nn.L1Loss(reduction='none')
            self.crit_bn = nn.L1Loss(reduction='none')
            self.crit_av = nn.L1Loss(reduction='none')
        # av distil loss
        if self.use_cmc_loss:
            self.crit_cmc = nn.L1Loss(reduction='none')
        if self.use_clm_loss:
            self.clm_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # warmup schedule
        self.warmup_epochs = self.args.get("warmup_epochs", -1)
        
        # add max epochs variable for new scheduler
        self.max_epochs = self.args.get("max_epochs", 50)
        self.pretrained_av_enc = \
            self.args["av_enc"].get("from_pretrained", False)
        self.finetune_av_enc = self.args["av_enc"].get("finetune", True)

        # multimodal parameters
        self.mmgpt = args["mmgpt"]
        self.tune_ffw = self.mmgpt.get("tune_ffw", True)
        self.use_lora = self.mmgpt.get("use_lora", False)
        
        # metrics
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.feature_name_map = {
            'T': 'Feature_t',
            'A': 'Feature_a',
            'V': 'Feature_v',
        }
        # self.args['device'] = 'cpu'

        # ulgm
        self.use_ulgm = self.args.get("use_ulgm", False)
        if self.use_ulgm:
            self.init_ulgm(args)
            self.ulgm_patience = self.args.get("ulgm_patience", 0)

    def _fusion_token_contrastive_loss(self, features, labels):
        """Soft supervised contrastive loss for regression sentiment labels."""
        batch_size = features.size(0)
        if batch_size < 2:
            return features.new_zeros(())

        temperature = max(float(self.contrast_temperature), 1e-6)
        sigma = max(float(self.contrast_label_sigma), 1e-6)

        z = F.normalize(features.float(), p=2, dim=1)
        y = labels.view(batch_size, -1).float()
        label_dist = torch.cdist(y, y, p=1)

        positive_weights = torch.exp(-(label_dist ** 2) / (2.0 * sigma ** 2))
        non_self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_weights = positive_weights * non_self_mask.float()
        row_sum = positive_weights.sum(dim=1, keepdim=True)
        valid_rows = row_sum.squeeze(1) > 1e-8

        if not torch.any(valid_rows):
            return features.new_zeros(())

        positive_weights = positive_weights / row_sum.clamp_min(1e-8)
        logits = torch.matmul(z, z.transpose(0, 1)) / temperature
        logits = logits.masked_fill(~non_self_mask, -1e9)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -(positive_weights * log_prob).sum(dim=1)
        return loss[valid_rows].mean()

    def _mean_prediction_np(self, x):
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.detach().float().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3:
            x = x.mean(axis=1)
        if x.ndim > 1:
            x = x.reshape(x.shape[0], -1).mean(axis=1)
        return x.astype(np.float32)

    def _bn_helpfulness_np(self, labels, task_logits, bn_logits, beta):
        labels = np.asarray(labels).reshape(-1).astype(np.float32)
        task_logits = np.asarray(task_logits).reshape(-1).astype(np.float32)
        bn_logits = np.asarray(bn_logits).reshape(-1).astype(np.float32)
        task_error = np.abs(task_logits - labels)
        bn_error = np.abs(bn_logits - labels)
        helpful = bn_error < task_error
        calib = (1.0 - beta) * task_logits + beta * bn_logits
        calib_error = np.abs(calib - labels)
        gain = task_error - calib_error
        return {
            "BNHelpfulRatio": float(np.mean(helpful)),
            "BNHelpful_CalibGain": float(np.mean(gain[helpful])) if np.any(helpful) else 0.0,
            "BNHarmful_CalibLoss": float(np.mean(-gain[~helpful])) if np.any(~helpful) else 0.0,
            "BNOracle_UpperGain": float(np.mean(np.maximum(task_error - bn_error, 0.0))),
            "Gap_task_bn": float(np.mean(np.abs(task_logits - bn_logits))),
            "Gap_calib_bn": float(np.mean(np.abs(calib - bn_logits))),
            "BNCalib_PredDelta": float(np.mean(np.abs(calib - task_logits))),
            "Task_MAE_diag": float(np.mean(task_error)),
            "BN_MAE_diag": float(np.mean(bn_error)),
            "Calib_MAE_diag": float(np.mean(calib_error)),
        }

    def _branch_diagnostics_np(self, labels, task_logits, bn_logits, text_logits=None, av_logits=None):
        labels = np.asarray(labels).reshape(-1).astype(np.float32)
        branches = {
            "task": np.asarray(task_logits).reshape(-1).astype(np.float32),
            "bn": np.asarray(bn_logits).reshape(-1).astype(np.float32),
        }
        if text_logits is not None:
            branches["text"] = np.asarray(text_logits).reshape(-1).astype(np.float32)
        if av_logits is not None:
            branches["av"] = np.asarray(av_logits).reshape(-1).astype(np.float32)

        n = min([len(labels)] + [len(v) for v in branches.values()])
        labels = labels[:n]
        branches = {k: v[:n] for k, v in branches.items()}
        errors = {k: np.abs(v - labels) for k, v in branches.items()}
        branch_names = list(branches.keys())
        error_stack = np.stack([errors[k] for k in branch_names], axis=1)
        best_idx = np.argmin(error_stack, axis=1)
        task_error = errors["task"]
        oracle_error = error_stack[np.arange(n), best_idx]

        metrics = {}
        for i, name in enumerate(branch_names):
            metrics[f"Branch_{name}_MAE"] = float(np.mean(errors[name]))
            metrics[f"BranchBest_{name}_ratio"] = float(np.mean(best_idx == i))

        metrics["BranchOracle_MAE"] = float(np.mean(oracle_error))
        metrics["BranchOracle_Gain"] = float(np.mean(task_error - oracle_error))
        metrics["BranchNonTaskBestRatio"] = float(np.mean(best_idx != branch_names.index("task")))
        if "bn" in errors:
            metrics["BNHelpfulRatio"] = float(np.mean(errors["bn"] < task_error))
        if "text" in errors:
            metrics["TextHelpfulRatio"] = float(np.mean(errors["text"] < task_error))
        if "av" in errors:
            metrics["AVHelpfulRatio"] = float(np.mean(errors["av"] < task_error))
        return metrics

    def _pearson_np(self, x, y):
        x = np.asarray(x).reshape(-1).astype(np.float64)
        y = np.asarray(y).reshape(-1).astype(np.float64)
        if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        return corr if np.isfinite(corr) else 0.0

    def _branch_occlusion_diagnostics_np(self, labels, task_logits, occluded_logits):
        """Compute identical representation-occlusion diagnostics for every task head."""
        labels = np.asarray(labels).reshape(-1).astype(np.float32)
        full = np.asarray(task_logits).reshape(-1).astype(np.float32)
        branches = ("fusion", "text", "av")
        masked = {
            name: np.asarray(occluded_logits[name]).reshape(-1).astype(np.float32)
            for name in branches
        }

        n = min([len(labels), len(full)] + [len(masked[name]) for name in branches])
        labels = labels[:n]
        full = full[:n]
        masked = {name: values[:n] for name, values in masked.items()}

        abs_changes = np.stack(
            [np.abs(full - masked[name]) for name in branches], axis=1
        ).astype(np.float32)
        normalized = abs_changes / np.maximum(abs_changes.sum(axis=1, keepdims=True), 1e-8)
        full_mae = float(np.mean(np.abs(full - labels)))
        full_corr = self._pearson_np(full, labels)

        metrics = {
            "Occlusion_full_MAE": full_mae,
            "Occlusion_full_Corr": full_corr,
        }
        for index, name in enumerate(branches):
            masked_mae = float(np.mean(np.abs(masked[name] - labels)))
            masked_corr = self._pearson_np(masked[name], labels)
            metrics[f"Occlusion_{name}_MeanAbsChange"] = float(abs_changes[:, index].mean())
            metrics[f"Occlusion_{name}_NormalizedSensitivity"] = float(
                normalized[:, index].mean()
            )
            metrics[f"Occlusion_{name}_MAE"] = masked_mae
            metrics[f"Occlusion_{name}_DeltaMAE"] = masked_mae - full_mae
            metrics[f"Occlusion_{name}_Corr"] = masked_corr
            metrics[f"Occlusion_{name}_DeltaCorr"] = full_corr - masked_corr

        arrays = {
            "occlusion_abs_change": abs_changes,
            "occlusion_normalized_sensitivity": normalized.astype(np.float32),
        }
        return metrics, arrays

    def _task_head_contribution_ratios_np(self, model):
        task_head = getattr(model.Model, "W_task", None)
        first_linear = None
        if isinstance(task_head, nn.Linear):
            first_linear = task_head
        elif isinstance(task_head, nn.Sequential):
            for module in task_head:
                if isinstance(module, nn.Linear):
                    first_linear = module
                    break
        if first_linear is None:
            return {}

        try:
            d_bn = int(self.args["mmgpt"]["d_mm"])
            d_text = int(self.args["mmgpt"]["d_mm"])
            d_av = int(self.args["av_enc"]["d_enc_out"])
        except Exception:
            return {}

        total_dim = d_bn + d_text + d_av
        weight = first_linear.weight.detach().abs().float().cpu().numpy()
        if weight.shape[1] < total_dim:
            return {}

        bn_score = float(weight[:, :d_bn].sum())
        text_score = float(weight[:, d_bn:d_bn + d_text].sum())
        av_score = float(weight[:, d_bn + d_text:d_bn + d_text + d_av].sum())
        total = max(bn_score + text_score + av_score, 1e-8)
        return {
            "Head_bn_ratio": bn_score / total,
            "Head_text_ratio": text_score / total,
            "Head_av_ratio": av_score / total,
        }

    def _diagnostic_dump_path(self, mode):
        prefix = self.dump_diagnostics_prefix
        seed = self.args.get("seed", self.args.get("cur_seed", "seed"))
        if not prefix:
            exp_name = self.args.get("exp_name", self.args.get("model_name", "msalm"))
            prefix = f"{self.args.dataset_name}_{exp_name}_{seed}_{mode}".lower()
        else:
            prefix = f"{prefix}_{seed}_{mode}".lower()
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix))
        return os.path.join(self.dump_diagnostics_dir, f"{safe_prefix}.npz")

    def _multimodal_loss(self, task_loss, bn_loss, av_loss, text_loss):
        return (
            task_loss
            + self.args.l_bn * bn_loss
            + self.args.l_av * av_loss
            + self.args.l_t * text_loss
        )

    def init_ulgm(self, args):
        self.feature_map = {
            'fusion': torch.zeros(
                args.train_samples,
                args.mmgpt.d_out,
                requires_grad=False
            ).to(args.device),
            'text': torch.zeros(
                args.train_samples,
                args.mmgpt.n_embd,
                requires_grad=False
            ).to(args.device),
            'av': torch.zeros(
                args.train_samples,
                args.av_enc.d_enc,
                requires_grad=False
            ).to(args.device),
            'bn': torch.zeros(
                args.train_samples,
                args.mmgpt.n_embd,
                requires_grad=False
            ).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.mmgpt.d_out, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.mmgpt.d_out, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.mmgpt.n_embd, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.mmgpt.n_embd, requires_grad=False).to(args.device),
            },
            'av': {
                'pos': torch.zeros(args.av_enc.d_enc, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.av_enc.d_enc, requires_grad=False).to(args.device),
            },
            'bn': {
                'pos': torch.zeros(args.mmgpt.n_embd, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.mmgpt.n_embd, requires_grad=False).to(args.device),
            }
        }

        self.dim_map = {
            'fusion': torch.tensor(args.mmgpt.n_embd).float(),
            'text': torch.tensor(args.mmgpt.n_embd).float(),
            'av': torch.tensor(args.av_enc.d_enc).float(),
            'bn': torch.tensor(args.mmgpt.n_embd).float(),
        }

        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'av': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'bn': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'task_logits': 'fusion',
            'text_logits': 'text',
            'av_logits': 'av',
            'bn_logits': 'bn'
        }

        self.tasks = ["task_logits", "av_logits", "text_logits", "bn_logits"]


    def do_train(self, model, dataloader, return_epoch_results=False):
        # optimizer configuration
        self.gamma = self.args.gamma
        self.args["dense_decay"] = False

        gpt_no_decay = [
            'bias',
            'LayerNorm.bias',
            'LayerNorm.weight',
            '_LlamaRMSNorm.weight'
        ]

        # Freeze all parameters
        model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

        # tune the cross attention layers
        if "gpt" in self.lm_flavor:
            trainable_list = [
                "alpha_1", "alpha_2", "ln_1", "ln_2", "attn",
                "bn_embedding"
            ]
            if self.mmgpt.use_lora:
                trainable_list.extend(
                    ["lora_c_fc", "lora_c_proj"]
                )
            else:
                # full fine-tuning
                trainable_list.extend(
                    ["c_fc", "c_proj"]
                )                     
        else:
            trainable_list = [
                "alpha_1", "alpha_2", "ln_1", "ln_2", "attn",
                "lora_gate_proj", "lora_up_proj", "lora_down_proj",
                "bn_embedding"
            ]

        if "gpt" in self.lm_flavor:
            # gpt
            for n, p in model.Model.lang_encoder.transformer.h.named_parameters():
                if any(s in n for s in trainable_list) and ("ca_layer" in n):
                    print(n)
                    p.requires_grad_(True)
            for n, p in model.Model.lang_encoder.transformer.wte.named_parameters():
                if any(s in n for s in trainable_list):
                    print(n)
                    p.requires_grad_(True)
        else:
            # llama
            for n, p in model.Model.lang_encoder.model.layers.named_parameters():
                if any(s in n for s in trainable_list) and ("ca_layer" in n):
                    print(n)
                    p.requires_grad_(True)
            for n, p in model.Model.lang_encoder.model.embed_tokens.named_parameters():
                if any(s in n for s in trainable_list):
                    print(n)
                    p.requires_grad_(True)

        # AV encoder tuning 
        model.Model.av_encoder.requires_grad_(True)
        # Task layer tuning
        if self.use_ulgm:
            model.Model.W_task_0.requires_grad_(True)
            model.Model.W_task_1.requires_grad_(True)
        else:
            model.Model.W_task.requires_grad_(True)
        model.Model.W_bn.requires_grad_(True)
        model.Model.W_text.requires_grad_(True)
        model.Model.W_av.requires_grad_(True)
        if getattr(model.Model, "contribution_gate", None) is not None:
            model.Model.contribution_gate.requires_grad_(True)
        if getattr(model.Model, "global_branch_logits", None) is not None:
            # Global learned-weight control: these three logits are the only
            # sample-independent aggregation parameters and must be unfrozen
            # after the trainer freezes the complete model above.
            model.Model.global_branch_logits.requires_grad_(True)
        if self.args.use_lnorm:
            model.Model.LN.requires_grad_(True)
        # model.Model.av_dec.requires_grad_(True)

        total_trainable = 0
        for n, p in model.named_parameters():
            if  p.requires_grad:
                print(n)
                total_trainable += p.numel()
        # Convert to millions and format the output
        total_trainable_millions = total_trainable / 1_000_000
        print(f"The total number of trainable parameters is {total_trainable_millions:.2f} M")
        # print(f"The totalnumber of trainable parameters is {total_trainable}")

        # seperate av params w and wout decay
        av_params = [
            (f"av_encoder.{n}", p)
            for n, p in model.Model.av_encoder.named_parameters()
        ]
        # av encoder params
        av_params_list = [f'Model.{n}' for n,_ in av_params]
        av_params_decay = \
            [p for n, p in av_params if not any(nd in n for nd in gpt_no_decay)]
        av_params_no_decay = \
            [p for n, p in av_params if any(nd in n for nd in gpt_no_decay)]

        # separate to params w and wout decay
        params_decay = []
        params_no_decay = []
        for n, p in model.named_parameters():
            # check if already in av_params_list
            if n not in av_params_list:
                print(n)
                if p.requires_grad:
                    if any(nd in n for nd in gpt_no_decay):
                        print(f"Using grad with no decay in {n}")
                        params_no_decay.append(p)
                    else:
                        print(f"Using grad with decay in {n}")
                        params_decay.append(p)
        
        # check if av_params + params are equal to total_trainable
        
        # Calculate the total number of elements
        tot_av_params_decay = sum(tensor.numel() for tensor in av_params_decay)
        tot_av_params_no_decay = sum(tensor.numel() for tensor in av_params_no_decay)
        tot_params_decay = sum(tensor.numel() for tensor in params_decay)
        tot_params_no_decay = sum(tensor.numel() for tensor in params_no_decay)
        assert total_trainable == (
                    tot_av_params_decay +
                    tot_av_params_no_decay +
                    tot_params_decay +
                    tot_params_no_decay
        )
        
        optimizer_grouped_parameters = [
            {
                'params': params_decay,
                'weight_decay': self.args.weight_decay_mmgpt,
                'lr': self.args.learning_rate_mmgpt,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )
            },
            {
                'params': params_no_decay,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate_mmgpt,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )

            },
            {
                'params': av_params_decay,
                'weight_decay': self.args.weight_decay_av,
                'lr': self.args.learning_rate_av,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )
            },
            {
                'params': av_params_no_decay,
                'weight_decay': 0.0,
                'lr': self.args.learning_rate_av,
                'betas': (
                    self.args.get('beta_1', 0.9),
                    self.args.get('beta_2', 0.999)
                )

            }
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters)

        ###########################################################################################
        ## new version of sceduler
        ###########################################################################################
        if self.warmup_epochs > 0:
            steps_per_epoch = int(len(dataloader["train"]) / self.args.update_epochs)
            warmup_steps = steps_per_epoch * self.warmup_epochs
            warm_scheduler = get_scheduler(
                optimizer,
                self.max_epochs,
                steps_per_epoch,
                warmup_steps,
            )
            print(f"Will be using warmup for {warmup_steps} steps")
            # warm_scheduler = \
            #     get_linear_schedule_with_warmup(optimizer, warmup_steps)

        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss', 'MAE'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        
        ###########################################################################################
        ## training loop
        ###########################################################################################
        while True:
            epochs += 1
            # train
            y_pred, y_true = [], []
            if self.use_ulgm:
                y_pred = {'fusion': [], 'av': [], 'text': [], 'bn': []}
                y_true = {'fusion': [], 'av': [], 'text': [], 'bn': []}
            
            model.train()
            # model.init_mmt()
            total_loss = 0.0
            total_aux_loss = 0.0
            train_loss = 0.0
            lm_loss = 0.0
            bn_total_loss = .0
            av_total_loss = .0
            text_total_loss = .0
            fusion_contrast_total_loss = .0
            # loss = .0
            left_epochs = self.args.update_epochs
            ids = []
            # aug_mix_ratio = self.args.get('aug_mix_ratio', 0.0)
            # aug_mix_steps = 0
            # print(model.Model.lang_encoder.model.embed_tokens[0].bn_embedding.bn_embedding)
            with tqdm(dataloader['train']) as td:
                step_counter = 0
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    
                    # idx-es for ulgm
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)
                    
                    # language modality handling
                    raw_text = batch_data['raw_text']
                    tokenized_inputs = model.Model.tokenizer(
                        raw_text, return_tensors="pt",
                        padding="max_length",
                        truncation=True
                    ).to(self.args.device)
                    text_ids = tokenized_inputs['input_ids']
                    
                    # 1 for valid positions and -1 for invalid --- Create binary mask
                    attention_mask = tokenized_inputs['attention_mask']
                    # prepa lm ids
                    # Shift the tensor to the left by 1
                    lm_text_tgt = torch.roll(text_ids, -1, dims=1)
                    # Replace the last element of each row with eos_token
                    lm_text_tgt[:, -1] = 0
                    
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    # lm_logits: (B, L, |V|)
                    # task_logits: (B, L, 1)
                    if self.use_bf16:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                           outputs = model(
                               text_ids,
                               audio,
                               vision,
                               attention_mask=attention_mask,
                            )
                           lm_logits = outputs['lm_logits']
                           task_logits = outputs['task_logits']
                           av_logits = outputs['av_logits']
                           bn_logits = outputs['bn_logits']
                           text_logits = outputs['text_logits']
                    else:
                        outputs = model(
                            text_ids,
                            audio,
                            vision,
                            attention_mask=attention_mask,
                        )
                        lm_logits = outputs['lm_logits']
                        task_logits = outputs['task_logits']
                        av_logits = outputs['av_logits']
                        bn_logits = outputs['bn_logits']
                        text_logits = outputs['text_logits']
                    av_logits = av_logits.squeeze(1)

                    # compute loss
                    loss = .0
                    B, L = text_ids.size()
                    if self.n_bn_fusion > 0 and self.modded_loss:
                        ###########################################################################
                        ## ULGM
                        if self.use_ulgm and epochs > self.ulgm_patience:
                            print("Should not be here")
                        else:
                            #######################################################################
                            ## mutlimodal task loss calculation
                            expanded_labels = labels.expand_as(task_logits)
                            if self.use_bf16:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    task_loss = self.criterion(task_logits, expanded_labels)
                                    task_loss = torch.mean(task_loss) # average over mini-batch
                            else:
                                task_loss = self.criterion(task_logits, expanded_labels)
                                task_loss = torch.mean(task_loss) # average over mini-batch
                            #######################################################################
                            ## bn fusion loss
                            bn_labels = labels.unsqueeze(1)
                            bn_labels = bn_labels.expand_as(bn_logits)
                            if self.use_bf16:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    bn_loss = self.crit_bn(bn_logits, bn_labels)
                                    bn_loss = torch.mean(bn_loss) # average over fusion tokens and mini-batch
                            else:
                                bn_loss = self.crit_bn(bn_logits, bn_labels)
                                # here we can manipulate each pf the `n_bn_fusion` tokens differently if we wish
                                bn_loss = torch.mean(bn_loss) # average over fusion tokens and mini-batch
                            #######################################################################
                            ## av loss
                            av_logits = av_logits.unsqueeze(1)
                            if self.use_bf16:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    av_loss = self.crit_av(av_logits, labels)
                                    av_loss = torch.mean(av_loss)
                            else:
                                av_loss = self.crit_av(av_logits, labels)
                                # here we can manipulate each pf the `n_bn_fusion` tokens differently if we wish
                                av_loss = torch.mean(av_loss) # average over fusion tokens and mini-batch
                            ###########################################################################
                            ## text loss
                            if self.use_bf16:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    text_loss = self.crit_text(text_logits, labels)
                                    text_loss = torch.mean(text_loss)
                            else:
                                text_loss = self.crit_text(text_logits, labels)
                                # here we can manipulate each pf the `n_bn_fusion` tokens differently if we wish
                                text_loss = torch.mean(text_loss) # average over fusion tokens and mini-batch
                            #######################################################################
                            ## cmc loss
                            if self.use_cmc_loss:
                                if self.use_bf16:
                                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                        cmc_loss = self.crit_cmc(av_logits, text_logits)
                                        cmc_loss = torch.mean(cmc_loss)
                                else:
                                    cmc_loss = self.crit_cmc(av_logits, text_logits)
                                    # here we can manipulate each pf the `n_bn_fusion` tokens differently if we wish
                                    cmc_loss = torch.mean(cmc_loss) # average over fusion tokens and mini-batch
                    else:
                        if self.args.dense:
                            # reweighted loss computation
                            # uniform weighting
                            if self.args.dense_uni:
                                uniform = \
                                    torch.ones(B, L,
                                            device=self.args.device,
                                            requires_grad=False
                                            )
                                norm_factor = \
                                    torch.sum(attention_mask, dim=1) + 1e-6
                                uniform = uniform / norm_factor.unsqueeze(1)
                                dense_mask = uniform * attention_mask
                            elif self.args.dense_lin_decay:
                                if self.n_bn_fusion > 0:
                                    L = self.n_bn_fusion
                                    # Linearly decaying weights for each batch (B, L)
                                    linear_weights_batch = \
                                        torch.linspace(1, 0, L, device=self.args.device).repeat(B, 1)
                                    row_sums = linear_weights_batch.sum(dim=1, keepdim=True)
                                    dense_mask = linear_weights_batch / row_sums
                                else:
                                    # Linearly decaying weights for each batch (B, L)
                                    linear_weights_batch = \
                                        torch.linspace(1, 0, L, device=self.args.device).repeat(B, 1)
                                    # Apply the binary mask
                                    linear_weights_batch = linear_weights_batch * attention_mask
                                    # import pdb; pdb.set_trace()
                                    # Renormalize so that each row sums to 1
                                    row_sums = linear_weights_batch.sum(dim=1, keepdim=True)
                                    dense_mask = linear_weights_batch / row_sums
                            else:
                                raise KeyError("No dense loss weighting. Pls check config")
                            # requires reduction = 'none', and manual averaging over B
                            #  time dimension is already averaged via reweighting
                            task_logits = task_logits.squeeze(2)
                            if self.n_bn_fusion > 0:
                                # keep only the BN encodings for the task
                                task_logits = task_logits[:, self.args.max_token_len:]
                            expanded_labels = labels.expand_as(task_logits)
                            
                            if self.use_bf16:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    task_loss = self.criterion(task_logits, expanded_labels)
                            else:
                                task_loss = self.criterion(task_logits, expanded_labels)
                            if self.args.reweight_last:
                                if self.n_bn_fusion > 0:
                                    dense_mask[-1] = 1.0
                                else:
                                    last_valid_indices = \
                                        torch.sum(attention_mask, dim=1).long() - 1
                                    # get last non_zero_heads: add correction term
                                    # last_mask = dense_mask[
                                    #     torch.arange(B, device=self.args.device),
                                    #     last_valid_indices,
                                    # ]
                                    # dense_mask = dense_mask * self.args.lam / (1 - last_mask)
                                    if self.args.dense_decay:
                                        if (epochs-1) >= len(self.lam):
                                            lam_all = self.lam[-1]
                                        else:
                                            lam_all = self.lam[epochs-1]
                                        dense_mask = dense_mask * lam_all
                                    else:
                                        dense_mask = dense_mask * self.args.lam
                                    dense_mask[
                                        torch.arange(B, device=self.args.device),
                                        last_valid_indices,
                                        ] = 1 #1 - self.args.lam
                            task_loss = torch.sum(task_loss * dense_mask) / B

                    
                    # distil loss
                    if (self.n_bn_fusion > 0) and self.modded_loss:
                        if self.use_ulgm and epochs > self.ulgm_patience:
                            pass
                        else:
                            if self.use_bf16:
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    # L = global_loss + BN_loss + AV_loss + Lang_loss
                                    loss = self._multimodal_loss(
                                        task_loss, bn_loss, av_loss, text_loss
                                    )
                                    if self.use_cmc_loss:
                                        loss += self.args.l_cmc * cmc_loss
                            else:
                                loss = self._multimodal_loss(
                                    task_loss, bn_loss, av_loss, text_loss
                                )
                                if self.use_cmc_loss:
                                    loss += self.args.l_cmc * cmc_loss
                                
                    else:
                        # loss here is the "ensemble" and weighted from all involved 
                        # timesteps in each method
                        loss = task_loss

                        if self.args.n_bn_fusion > 0:
                            first_logits = task_logits[:, 0]
                            last_logits = task_logits[:, -1]
                        else:
                            last_valid_indices = \
                                        torch.sum(attention_mask, dim=1).long() - 1
                            last_logits = task_logits[
                                torch.arange(B, device=self.args.device),
                                last_valid_indices
                            ]
                            first_logits = task_logits[:, 0]
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 
                                distil_loss = \
                                    self.distil_crit(last_logits, first_logits)
                                loss += self.w_distil * distil_loss
                        else:
                            distil_loss = \
                                self.distil_crit(last_logits, first_logits)
                            loss += self.w_distil * distil_loss
                    
                    if (self.n_bn_fusion > 0) and self.modded_loss:
                        pass
                    else:
                        # av distil loss
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 
                                av_distil_loss_L = \
                                    self.distil_av_crit_xL(last_logits, av_logits)
                                av_distil_loss_0 = \
                                    self.distil_av_crit_x0(first_logits, av_logits)
                                loss += \
                                    self.w_av_distil * (av_distil_loss_0 + av_distil_loss_L)
                        else:
                            av_distil_loss_L = \
                                self.distil_av_crit_xL(last_logits, av_logits)
                            av_distil_loss_0 = \
                                self.distil_av_crit_x0(first_logits, av_logits)
                            loss += \
                                self.w_av_distil * (av_distil_loss_0 + av_distil_loss_L)
                    # clm loss
                    clm_loss = .0
                    if self.use_clm_loss:
                        # compute lm loss only on non-masked tokens
                        if self.n_bn_fusion > 0:
                            # (B, L+n, V) -> (B, L, V)
                            # print(f"computing clm loss")
                            lm_logits = lm_logits[:, :self.args.max_token_len, :].contiguous()
                        B, L, V = lm_logits.shape
                        lm_logits = lm_logits.view(B*L, V)
                        lm_text_tgt[~(attention_mask.bool())] = -1  # ignore index
                        lm_text_tgt = lm_text_tgt.reshape(B*L)
                        # ignore_idx = -1
                        if self.use_bf16:
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                clm_loss = self.clm_criterion(lm_logits, lm_text_tgt)
                                w_gamma = self.gamma
                                loss += w_gamma * clm_loss
                        else:
                            clm_loss = self.clm_criterion(lm_logits, lm_text_tgt)
                            w_gamma = self.gamma
                            # total_loss
                            loss += w_gamma * clm_loss

                    fusion_contrast_loss = .0
                    if (
                        self.use_fusion_contrast
                        and self.lambda_fusion_contrast > 0
                        and not (self.use_ulgm and epochs > self.ulgm_patience)
                    ):
                        fusion_features = outputs.get("Feature_bn", None)
                        if fusion_features is not None:
                            fusion_contrast_loss = self._fusion_token_contrastive_loss(
                                fusion_features, labels
                            )
                            loss += self.lambda_fusion_contrast * fusion_contrast_loss

                    if self.use_bf16:
                        self.scaler.scale(loss).backward()
                    else:
                        # backward
                        loss.backward()
                    
                    # store results
                    total_loss += loss.item()
                    if self.use_ulgm and epochs > self.ulgm_patience:
                        # update features
                        if self.use_bf16:
                            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            f_fusion = outputs['Feature_f'].to(torch.float).detach()
                            f_text = outputs['Feature_t'].to(torch.float).detach()
                            f_av = outputs['Feature_av'].to(torch.float).detach()
                            f_bn = outputs['Feature_bn'].to(torch.float).detach()
                            if epochs > self.args.update_labels_patience:
                                self.update_labels(
                                    f_fusion, f_text, f_av, f_bn, epochs, indexes, outputs
                                )

                            self.update_features(f_fusion, f_text, f_av, f_bn, indexes)
                            self.update_centers()
                        else:
                            f_fusion = outputs['Feature_f'].detach()
                            f_text = outputs['Feature_t'].detach()
                            f_av = outputs['Feature_av'].detach()
                            f_bn = outputs['Feature_bn'].detach()
                            if epochs > self.args.update_labels_patience:
                                self.update_labels(
                                    f_fusion, f_text, f_av, f_bn, epochs, indexes, outputs
                                )

                            self.update_features(f_fusion, f_text, f_av, f_bn, indexes)
                            self.update_centers()
                    else:
                        train_loss += task_loss.item()
                        if self.args.use_clm:
                            lm_loss += clm_loss.item()
                        if self.modded_loss:
                            bn_total_loss += bn_loss.item()
                            av_total_loss += av_loss.item()
                            text_total_loss += text_loss.item()
                        if torch.is_tensor(fusion_contrast_loss):
                            fusion_contrast_total_loss += fusion_contrast_loss.item()
                    y_pred.append(task_logits.to(torch.float32).cpu().detach())
                    # put expanded_labels here
                    y_true.append(expanded_labels.to(torch.float32).cpu().detach())
                    
                    if not left_epochs:
                        if self.use_bf16:
                             # grad clip
                            if self.args.grad_clip != -1.0:
                                # Unscales the gradients of optimizer's assigned params in-place
                                self.scaler.unscale_(optimizer)
                                    # TODO: mmgpt might not require grad clipping
                                nn.utils.clip_grad_value_(
                                    [param for param in params_decay],
                                    self.args.grad_clip
                                )
                                nn.utils.clip_grad_value_(
                                    [param for param in params_no_decay],
                                    self.args.grad_clip
                                )
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            # grad clip
                            if self.args.grad_clip != -1.0:
                            # TODO: mmgpt might not require grad clipping
                                nn.utils.clip_grad_value_(
                                    [param for param in params_decay],
                                    self.args.grad_clip
                                )
                                nn.utils.clip_grad_value_(
                                    [param for param in params_no_decay],
                                    self.args.grad_clip
                                )
                            optimizer.step()
                        warm_scheduler.step()
                        if step_counter <=3:
                            my_dict = optimizer.state_dict()
                            # for k in my_dict['param_groups']:
                            #     print(k['lr'])
                        step_counter += 1

                        left_epochs = self.args.update_epochs
                # trick for last batch update
                if not left_epochs:
                    # update
                    if self.use_bf16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()  
                    # optimizer.step()
                    warm_scheduler.step()
            
            train_loss = train_loss / len(dataloader['train'])
            total_loss = total_loss / len(dataloader['train'])
            bn_total_loss = bn_total_loss / len(dataloader['train'])
            av_total_loss = av_total_loss / len(dataloader['train'])
            text_total_loss = text_total_loss / len(dataloader['train'])
            fusion_contrast_total_loss = fusion_contrast_total_loss / len(dataloader['train'])
            lm_loss = lm_loss / len(dataloader['train'])
            # if self.modded_loss:
            #     bn_total_loss = bn_total_loss / len(dataloader['train'])
            # mse_loss = mse_loss / len(dataloader['train'])
            # barlow_loss = barlow_loss / len(dataloader['train'])
            total_aux_loss = total_aux_loss / len(dataloader['train'])

            if self.use_ulgm and epochs > self.ulgm_patience:
                for m in self.tasks:
                    # if m == 'bn_logits':
                    #     import pdb; pdb.set_trace()
                    pred, true = \
                        torch.cat(y_pred[self.name_map[m]]), torch.cat(y_true[self.name_map[m]])
                    train_results = self.metrics(pred, true)
                    logger.info(
                        f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >>" \
                        f" loss: {round(total_loss, 4)}" \
                        f">> {dict_to_str(train_results)}"
                    )
            else:
                pred, true = torch.cat(y_pred), torch.cat(y_true)
                train_results = self.metrics(pred, true)
                logger.info(
                    f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] >>" \
                    f" loss: {round(train_loss, 4)} {dict_to_str(train_results)}" \
                    f" clm loss: {round(lm_loss, 4)}" \
                    f" total loss: {round(total_loss, 4)}" \
                    f" bn loss: {round(bn_total_loss, 4)}" \
                    f" av loss: {round(av_total_loss, 4)}" \
                    f" text loss: {round(text_total_loss, 4)}" \
                    + (
                        f" fusion contrast loss: {round(fusion_contrast_total_loss, 4)}"
                        if self.use_fusion_contrast else ""
                    )
                )
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # if self.warmup_epochs < epochs:
            #     scheduler.step(val_results['Loss'])
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None
            # if epochs - best_epoch >= 15:
            #     return epoch_results if return_epoch_results else None

    
    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        print(f"Model alphas are")
        if 'gpt' in self.lm_flavor:
            for n, p in model.Model.lang_encoder.transformer.h.named_parameters():
                if "alpha" in n:
                    print(f"{n} is: {p}")
        else:
            for n, p in model.Model.lang_encoder.model.layers.named_parameters():
                if "alpha" in n:
                    print(f"{n} is: {p}")
        y_pred, y_true = [], []
        if self.use_ulgm:
            y_pred = {'fusion': [], 'av': [], 'text': [], 'bn': []}
            y_true = {'fusion': [], 'av': [], 'text': [], 'bn': []}
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        diag_scalar = {}
        ids = []
        diagnostic_mode_matches = str(mode).upper() == self.dump_diagnostics_mode
        should_dump_branch_occlusion = (
            self.dump_branch_occlusion_diagnostics and diagnostic_mode_matches
        )
        should_dump_diagnostics = (
            (self.dump_baseline_diagnostics or should_dump_branch_occlusion)
            and diagnostic_mode_matches
        )
        diagnostic_dump = None
        if should_dump_diagnostics:
            diagnostic_dump = {
                "feature_bn": [],
                "labels": [],
                "task_logits": [],
                "bn_logits": [],
                "text_logits": [],
                "av_logits": [],
                "contribution_weights": [],
                "occlusion_fusion_logits": [],
                "occlusion_text_logits": [],
                "occlusion_av_logits": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    # language modality handling
                    raw_text = batch_data['raw_text']
                    tokenized_inputs = model.Model.tokenizer(
                        raw_text, return_tensors="pt",
                        padding="max_length",
                        truncation=True
                    ).to(self.args.device)
                    text_ids = tokenized_inputs['input_ids']
                    attention_mask = tokenized_inputs['attention_mask']
                    # idx-es for ulgm
                    indexes = batch_data['index'].view(-1)
                    cur_id = batch_data['id']
                    ids.extend(cur_id)
                    
                    # task_binary_mask = (attention_mask == 1).int()
                    last_valid_indices = \
                            torch.sum(attention_mask, dim=1).long() - 1

                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)

                    
                    if self.use_ulgm:
                        # logit - loss - calculation
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                outputs = \
                                    model(
                                        text_ids,
                                        audio,
                                        vision,
                                        attention_mask=attention_mask,
                                    )
                                loss = self.weighted_loss(outputs['task_logits'], labels)
                        else:
                            outputs = \
                                model(
                                    text_ids,
                                    audio,
                                    vision,
                                    attention_mask=attention_mask,
                                )
                            loss = self.weighted_loss(outputs['task_logits'], labels)
                        # gather predictions
                        
                        y_pred["fusion"].append(
                            outputs["task_logits"].to(torch.float).cpu()
                        )
                        y_true["fusion"].append(
                            labels
                        )
                    elif self.modded_loss:
                        # The normal evaluation path keeps the historical
                        # bfloat16 autocast behavior.  Frozen-gate diagnostics
                        # can request FP32 so that task logits, branch logits,
                        # and mixture weights satisfy the exact mixture
                        # reconstruction check.
                        eval_context = (
                            nullcontext()
                            if self.args.get("diagnostic_fp32", False)
                            else torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                        )
                        with eval_context:
                            outputs = model(
                                text_ids,
                                audio,
                                vision,
                                attention_mask=attention_mask,
                                return_branch_occlusion=should_dump_branch_occlusion,
                            )
                            task_logits = outputs['task_logits']
                            contribution_weights = outputs.get("Contribution_weights", None)
                            if contribution_weights is not None and self.dump_baseline_diagnostics:
                                diag_scalar.setdefault("Head_bn_ratio", []).append(
                                    contribution_weights[:, 0].detach().float().mean().item()
                                )
                                diag_scalar.setdefault("Head_text_ratio", []).append(
                                    contribution_weights[:, 1].detach().float().mean().item()
                                )
                                diag_scalar.setdefault("Head_av_ratio", []).append(
                                    contribution_weights[:, 2].detach().float().mean().item()
                                )
                            # av_logits = outputs['av_logits']
                            # av_logits.squeeze_(1)                        
                    elif self.aux or self.args.get("av_distil", False):
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                lm_logits, task_logits, _, bn_logits = \
                                    model(
                                        text_ids,
                                        audio,
                                        vision,
                                        attention_mask=attention_mask,
                                    )
                        else:
                            lm_logits, task_logits, _, _ = \
                                model(
                                    text_ids,
                                    audio,
                                    vision,
                                    attention_mask=attention_mask,
                                )
                    else:
                        lm_logits, task_logits = model(text_ids, audio, vision)

                    B, L = text_ids.shape
                    
                    
                    
                    # Gather the last non-zero logits using the corrected indices
                    if self.modded_loss:
                        pass
                    else:
                        if self.n_bn_fusion > 0:
                            last_non_zero_logits = task_logits[:, -1]
                        else:
                            last_non_zero_logits = \
                                task_logits[
                                    torch.arange(B, device=self.args.device),
                                    last_valid_indices
                                ]

                    # SOS: here we evaluate only on the last logit as in vanilla setups
                    # B, _ = labels.shape
                    if self.use_ulgm:
                        pass
                    elif self.modded_loss:
                        ##### task loss
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                loss = self.criterion(task_logits.view(-1), labels.view(-1))
                                loss = torch.mean(loss)
                        else:
                            loss = self.criterion(task_logits.view(-1), labels.view(-1))
                            loss = torch.mean(loss)
                        if diagnostic_dump is not None:
                            feature_bn = outputs.get("Feature_bn", None)
                            bn_logits_for_dump = outputs.get("bn_logits", None)
                            text_logits_for_dump = outputs.get("text_logits", None)
                            av_logits_for_dump = outputs.get("av_logits", None)
                            contribution_weights_for_dump = outputs.get("Contribution_weights", None)
                            if feature_bn is not None and bn_logits_for_dump is not None:
                                diagnostic_dump["feature_bn"].append(
                                    feature_bn.detach().float().cpu().numpy()
                                )
                                diagnostic_dump["labels"].append(
                                    labels.detach().float().cpu().numpy()
                                )
                                diagnostic_dump["task_logits"].append(
                                    task_logits.detach().float().cpu().numpy()
                                )
                                diagnostic_dump["bn_logits"].append(
                                    bn_logits_for_dump.detach().float().cpu().numpy()
                                )
                                if text_logits_for_dump is not None:
                                    diagnostic_dump["text_logits"].append(
                                        text_logits_for_dump.detach().float().cpu().numpy()
                                    )
                                if av_logits_for_dump is not None:
                                    diagnostic_dump["av_logits"].append(
                                        av_logits_for_dump.detach().float().cpu().numpy()
                                    )
                                if contribution_weights_for_dump is not None:
                                    diagnostic_dump["contribution_weights"].append(
                                        contribution_weights_for_dump.detach().float().cpu().numpy()
                                    )
                                occlusion_logits = outputs.get("Branch_occlusion_logits", None)
                                if occlusion_logits is not None:
                                    for branch in ("fusion", "text", "av"):
                                        diagnostic_dump[f"occlusion_{branch}_logits"].append(
                                            occlusion_logits[branch].detach().float().cpu().numpy()
                                        )
                    else:
                        if self.use_bf16:
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                loss = self.criterion(last_non_zero_logits.view(-1), labels.view(-1))
                                loss = torch.mean(loss)
                        else:
                            loss = self.criterion(last_non_zero_logits.view(-1), labels.view(-1))
                            loss = torch.mean(loss)
                    eval_loss += loss.item()
                    
                    # gather predictions
                    if self.use_ulgm:
                        pass
                    elif self.modded_loss:
                        y_pred.append(task_logits.to(torch.float32).cpu().detach())
                        y_true.append(labels.cpu())
                    else:    
                        y_pred.append(last_non_zero_logits.to(torch.float32).cpu().detach())
                        y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        if self.use_ulgm:
            pred, true = torch.cat(y_pred["fusion"]), torch.cat(y_true["fusion"])
            eval_results = self.metrics(pred, true)
            eval_results["Loss"] = round(eval_loss, 4)
            logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        else:
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            eval_results["Loss"] = round(eval_loss, 4)
            for k, values in diag_scalar.items():
                if values:
                    eval_results[k] = round(float(np.mean(values)), 4)
            if diagnostic_dump is not None and diagnostic_dump["feature_bn"]:
                os.makedirs(self.dump_diagnostics_dir, exist_ok=True)
                dump_arrays = {
                    "feature_bn": np.concatenate(diagnostic_dump["feature_bn"], axis=0),
                    "labels": np.concatenate(diagnostic_dump["labels"], axis=0),
                    "task_logits": np.concatenate(diagnostic_dump["task_logits"], axis=0),
                    "bn_logits": np.concatenate(diagnostic_dump["bn_logits"], axis=0),
                }
                dump_arrays["bn_logits_mean"] = self._mean_prediction_np(dump_arrays["bn_logits"])
                if diagnostic_dump["text_logits"]:
                    dump_arrays["text_logits"] = np.concatenate(diagnostic_dump["text_logits"], axis=0)
                    dump_arrays["text_logits_mean"] = self._mean_prediction_np(dump_arrays["text_logits"])
                if diagnostic_dump["av_logits"]:
                    dump_arrays["av_logits"] = np.concatenate(diagnostic_dump["av_logits"], axis=0)
                    dump_arrays["av_logits_mean"] = self._mean_prediction_np(dump_arrays["av_logits"])
                head_diag = {}
                if self.dump_baseline_diagnostics:
                    if diagnostic_dump["contribution_weights"]:
                        dump_arrays["contribution_weights"] = np.concatenate(
                            diagnostic_dump["contribution_weights"], axis=0
                        )
                        head_diag = {
                            "Head_bn_ratio": float(np.mean(dump_arrays["contribution_weights"][:, 0])),
                            "Head_text_ratio": float(np.mean(dump_arrays["contribution_weights"][:, 1])),
                            "Head_av_ratio": float(np.mean(dump_arrays["contribution_weights"][:, 2])),
                        }
                    else:
                        head_diag = self._task_head_contribution_ratios_np(model)
                    for k, v in head_diag.items():
                        dump_arrays[k] = np.asarray([v], dtype=np.float32)

                    bn_diag = self._bn_helpfulness_np(
                        dump_arrays["labels"],
                        self._mean_prediction_np(dump_arrays["task_logits"]),
                        dump_arrays["bn_logits_mean"],
                        self.dump_bn_calibration_beta,
                    )
                    branch_diag = self._branch_diagnostics_np(
                        dump_arrays["labels"],
                        self._mean_prediction_np(dump_arrays["task_logits"]),
                        dump_arrays["bn_logits_mean"],
                        dump_arrays.get("text_logits_mean", None),
                        dump_arrays.get("av_logits_mean", None),
                    )
                    for diagnostics in (bn_diag, branch_diag, head_diag):
                        for k, v in diagnostics.items():
                            eval_results[k] = round(float(v), 4)
                    eval_results["BNDiag_beta"] = round(float(self.dump_bn_calibration_beta), 4)

                if should_dump_branch_occlusion:
                    occlusion_keys = [
                        f"occlusion_{branch}_logits" for branch in ("fusion", "text", "av")
                    ]
                    if all(diagnostic_dump[key] for key in occlusion_keys):
                        occluded_logits = {}
                        for branch in ("fusion", "text", "av"):
                            key = f"occlusion_{branch}_logits"
                            dump_arrays[key] = np.concatenate(diagnostic_dump[key], axis=0)
                            occluded_logits[branch] = self._mean_prediction_np(dump_arrays[key])
                        occlusion_diag, occlusion_arrays = self._branch_occlusion_diagnostics_np(
                            dump_arrays["labels"],
                            self._mean_prediction_np(dump_arrays["task_logits"]),
                            occluded_logits,
                        )
                        dump_arrays.update(occlusion_arrays)
                        for k, v in occlusion_diag.items():
                            dump_arrays[k] = np.asarray([v], dtype=np.float32)
                            eval_results[k] = round(float(v), 4)
                    else:
                        logger.warning(
                            "Branch occlusion diagnostics were requested but masked logits "
                            "were not returned by the model."
                        )

                dump_path = self._diagnostic_dump_path(mode)
                np.savez_compressed(dump_path, **dump_arrays)
                logger.info(f"{mode}-({self.args.model_name}) diagnostic dump saved to {dump_path}")
            elif should_dump_diagnostics:
                logger.warning(
                    f"{mode}-({self.args.model_name}) diagnostic dump requested but Feature_bn/bn_logits were not collected. "
                    "Check that modded_loss=True and n_bn_fusion>0."
                )
            logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results

    def do_test_head(self, backbone, model, dataloader, mode="VAL", modal="T"):
        backbone.eval()
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    last_h = backbone(text, audio, vision)[self.feature_name_map[modal]]
                    outputs = model(last_h)

                    loss = self.criterion(outputs, labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}-{modal}) >> {dict_to_str(eval_results)}")

        return eval_results
