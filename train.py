import argparse
import copy
import json
import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
import csv
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW
from tensorboardX import SummaryWriter


from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from args import get_train_test_args


from tqdm import tqdm

def prepare_eval_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["id"] = []
    for i in tqdm(range(len(tokenized_examples["input_ids"]))):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["id"].append(dataset_dict["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples



def prepare_train_data(dataset_dict, tokenizer):
    tokenized_examples = tokenizer(dataset_dict['question'],
                                   dataset_dict['context'],
                                   truncation="only_second",
                                   stride=128,
                                   max_length=384,
                                   return_overflowing_tokens=True,
                                   return_offsets_mapping=True,
                                   padding='max_length')
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples['id'] = []
    inaccurate = 0
    for i, offsets in enumerate(tqdm(offset_mapping)):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answer = dataset_dict['answer'][sample_index]
        # Start/end character index of the answer in the text.
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        tokenized_examples['id'].append(dataset_dict['id'][sample_index])
        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)
            # assertion to check if this checks out
            context = dataset_dict['context'][sample_index]
            offset_st = offsets[tokenized_examples['start_positions'][-1]][0]
            offset_en = offsets[tokenized_examples['end_positions'][-1]][1]
            if context[offset_st : offset_en] != answer['text'][0]:
                inaccurate += 1

    total = len(tokenized_examples['id'])
    print(f"Preprocessing not completely accurate for {inaccurate}/{total} instances")
    return tokenized_examples



def read_and_process(args, tokenizer, dataset_dict, dir_name, dataset_name, split):
    #TODO: cache this if possible
    cache_path = f'{dir_name}/{dataset_name}_encodings.pt'
    if os.path.exists(cache_path) and not args.recompute_features:
        tokenized_examples = util.load_pickle(cache_path)
    else:
        if split=='train':
            tokenized_examples = prepare_train_data(dataset_dict, tokenizer)
        else:
            tokenized_examples = prepare_eval_data(dataset_dict, tokenizer)
        util.save_pickle(tokenized_examples, cache_path)
    return tokenized_examples



#TODO: use a logger, use tensorboard
class Trainer():
    def __init__(self, args, log):
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.device = args.device
        self.eval_every = args.eval_every
        self.path = os.path.join(args.save_dir, 'checkpoint')
        self.num_visuals = args.num_visuals
        self.save_dir = args.save_dir
        self.log = log
        self.visualize_predictions = args.visualize_predictions
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.device = args.device

    def save(self, model):
        model.save_pretrained(self.path)

    # There is a difference between meta-learning evaluation and final prediction evaluation
    # We won't really do eval during training for meta-learning
    # This is final prediction evaluation
    def evaluate(self, model, data_loader, data_dict, return_preds=False, split='validation'):
        device = self.device

        model.eval()
        pred_dict = {}

        all_start_logits_meta = []
        all_end_logits_meta = []
        # with torch.no_grad(), \
        with tqdm(total=len(data_loader.dataset)) as progress_bar:
            for task_batch in data_loader:
                # Change "parallel" weights for each task based on support
                for task in task_batch:
                    batch = task["support"]
                    for key in batch:
                        batch[key] = batch[key].to(self.device)
                        batch[key] = batch[key].squeeze(0)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    batch_size = len(input_ids)
                    # Adapt theta for meta-learning
                    # Change weights based on forward from "support"
                    alpha = 0.4
                    weights = torch.nn.Parameter(torch.clone(model.qa_outputs.weight))
                    model.qa_outputs.weight = weights
                    out = model.forward(**batch)
                    loss = out[0]
                    grad = torch.autograd.grad(loss, weights, create_graph=True)[0]
                    meta_weight = weights - alpha*grad
                    meta_weight = meta_weight.cpu().detach()
                    batch = task["query"]
                    for key in batch:
                        batch[key] = batch[key].to(self.device)
                        batch[key] = batch[key].squeeze(0)
                    for i in range(batch["input_ids"].shape[0]//10):
                        input_ids = batch["input_ids"][10*i:10*i+10]
                        attention_mask = batch["attention_mask"][10 * i:10 * i + 10]
                        # Outputs from regular forward
                        model.qa_outputs.weight = torch.nn.Parameter(meta_weight)
                        outputs_meta = model(input_ids, attention_mask=attention_mask)
                        # Forward
                        start_logits_meta, end_logits_meta = outputs_meta.start_logits, outputs_meta.end_logits
                        # TODO: compute loss

                        all_start_logits_meta.append(start_logits_meta.cpu().detach().numpy())
                        all_end_logits_meta.append(end_logits_meta.cpu().detach().numpy())
                    # progress_bar.update(batch_size)

      

        start_logits = torch.cat(all_start_logits_meta)
        end_logits = torch.cat(all_end_logits_meta)
        preds_meta, confidence_meta = util.postprocess_qa_predictions(data_dict,
                                                data_loader.dataset.encodings,
                                                (start_logits, end_logits))
        # Pick between preds_regular and preds_meta based on metric of confidence
        preds = {}
        for id in preds_meta:
            c_m = confidence_meta[id]
         
            preds[id] = preds_meta[id]

        if split == 'validation':
            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]
        else:
            results_list = [('F1', -1.0),
                            ('EM', -1.0)]
        results = OrderedDict(results_list)
        if return_preds:
            return preds, results
        return results

    def inner_loop(self, qa, model: DistilBertForQuestionAnswering):
        alpha = 0.3     # Learning rate, default from MAML code
        params = torch.clone(model.qa_outputs.weight)
        params = torch.nn.Parameter(params)
        model.qa_outputs.weight = params
        out = model(**qa)
        loss = out[0]
        grad = torch.autograd.grad(loss, params, create_graph=True)[0]
        params.requires_grad = False
        params -= alpha*grad
        return params

    def outer_step(self, task_batch, model: DistilBertForQuestionAnswering):
        outer_loss_batch = []
        weight_orig = copy.copy(model.qa_outputs.weight)
        for task in task_batch:
            print("FINISHED 1 TASK")
            qa_support, qa_query = task["support"], task["query"]
            for key in qa_support:
                qa_support[key] = qa_support[key].to(self.device)
                qa_support[key] = qa_support[key].squeeze(0)
            for key in qa_query:
                qa_query[key] = qa_query[key].to(self.device)
                qa_query[key] = qa_query[key].squeeze(0)
            params = self.inner_loop(qa_support, model)
            # outputs = model(input_ids, attention_mask=attention_mask,
            #                 start_positions=start_positions,
            #                 end_positions=end_positions)
            # Overwrite the model linear weights with params, assume inner_loop returns the weights of last layer
            model.qa_outputs.weight = torch.nn.Parameter(params)
            out = model(**qa_query)
            loss = out[0]
            outer_loss_batch.append(loss)
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        model.qa_outputs.weight = weight_orig
        return outer_loss

    def train(self, model: DistilBertForQuestionAnswering, train_dataloader):
        device = self.device
        model.to(device)
        optim = AdamW(model.parameters(), lr=self.lr)
        global_idx = 0
        best_scores = {'F1': -1.0, 'EM': -1.0}
        tbx = SummaryWriter(self.save_dir)

        # Freeze all layers except the "parallel" layer for metalearning
        # print(next(model.modules()))
        # model.distilbert
        model.distilbert.requires_grad_(False)
        model.qa_outputs.requires_grad_(False)
        # Check
        # model.distilbert.transformer.layer[0].attention.q_lin.weight.requires_grad

        for epoch_num in range(self.num_epochs):
            # preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
            # import pdb
            # pdb.set_trace()
            self.log.info(f'Epoch: {epoch_num}')
            with torch.enable_grad(), tqdm(total=len(train_dataloader.dataset)) as progress_bar:
                for task_batch in train_dataloader:
                    optim.zero_grad()
                    model.train()
# from parallel_model import ParallelModel
# m = ParallelModel.from_pretrained("distilbert-base-uncased")
# batch = task_batch
# input_ids = batch['input_ids'].to(device)
# attention_mask = batch['attention_mask'].to(device)
# start_positions = batch['start_positions'].to(device)
# end_positions = batch['end_positions'].to(device)
# outputs = m.forward_p(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
# loss = outputs[0]
# loss.backward()
#                     model.parallel.requires_grad_(False)
                    loss = self.outer_step(task_batch, model)
                    loss.backward()
                    optim.step()

            self.save(model)
        return 0

                    # progress_bar.update(len(input_ids))
                    # progress_bar.set_postfix(epoch=epoch_num, NLL=loss.item())
                    # tbx.add_scalar('train/NLL', loss.item(), global_idx)
                    # if (global_idx % self.eval_every) == 0:
                    #     self.log.info(f'Evaluating at step {global_idx}...')
                    #     preds, curr_score = self.evaluate(model, eval_dataloader, val_dict, return_preds=True)
                    #     results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in curr_score.items())
                    #     self.log.info('Visualizing in TensorBoard...')
                    #     for k, v in curr_score.items():
                    #         tbx.add_scalar(f'val/{k}', v, global_idx)
                    #     self.log.info(f'Eval {results_str}')
                    #     if self.visualize_predictions:
                    #         util.visualize(tbx,
                    #                        pred_dict=preds,
                    #                        gold_dict=val_dict,
                    #                        step=global_idx,
                    #                        split='val',
                    #                        num_visuals=self.num_visuals)
                    #     if curr_score['F1'] >= best_scores['F1']:
                    #         best_scores = curr_score
                    #         self.save(model)
                    # global_idx += 1
        # return best_scores

def get_dataset(args, datasets, data_dir, tokenizer, split_name, test_datasets=None, eval_dir=None):
    """
    Returns 2 lists:
    1) List of QADatasets
    2) List of dataset_dict's
    Size of both = num_datasets
    """
    datasets = datasets.split(',')
    dataset_name=''
    dataset_dict = {}
    all_data_encodings = []
    # Instead of merging datasets, just split them
    for dataset in datasets:
        dataset_name += f'_{dataset}'
        current_dataset_dict = util.read_squad(f'{data_dir}/{dataset}')
        data_encodings = read_and_process(args, tokenizer, current_dataset_dict, data_dir, dataset_name, "train")
        all_data_encodings.append(data_encodings)
        dataset_dict = util.merge(dataset_dict, current_dataset_dict)
    # Means we are in test mode!
    if test_datasets is not None:
        test_datasets = test_datasets.split(',')
        dataset_name = ''
        dataset_dict = {}
        test_data_encodings = []
        # Instead of merging datasets, just split them
        for dataset in test_datasets:
            dataset_name += f'_{dataset}'
            current_dataset_dict = util.read_squad(f'{eval_dir}/{dataset}')
            data_encodings = read_and_process(args, tokenizer, current_dataset_dict, data_dir, dataset_name, split_name)
            test_data_encodings.append(data_encodings)
            dataset_dict = util.merge(dataset_dict, current_dataset_dict)
        return util.QADatasets(all_data_encodings, num_support=args.num_support, num_query=args.num_query, test_encodings=test_data_encodings), dataset_dict

    return util.QADatasets(all_data_encodings, num_support=args.num_support, num_query=args.num_query), dataset_dict

def main():
    # define parser and arguments
    args = get_train_test_args()

    util.set_seed(args.seed)
    # model = ParallelModel.from_pretrained("distilbert-base-uncased")
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
    # TODO: ADD THIS BACK ON AZURE
    model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if args.do_train:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        args.save_dir = util.get_save_dir(args.save_dir, args.run_name)
        log = util.get_logger(args.save_dir, 'log_train')
        log.info(f'Args: {json.dumps(vars(args), indent=4, sort_keys=True)}')
        log.info("Preparing Training Data...")
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        trainer = Trainer(args, log)
        train_dataset, _ = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, 'train')
        log.info("Preparing Validation Data...")
        # val_dataset, val_dict  = get_dataset(args, args.train_datasets, args.val_dir, tokenizer, 'val')
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batch_size,
                                sampler=RandomSampler(train_dataset))
        # val_loader = DataLoader(dataset=val_dataset,
        #                         batch_size=args.batch_size,
        #                         sampler=SequentialSampler(val_dataset))
        best_scores = trainer.train(model, train_loader)
    if args.do_eval:
        args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        log = util.get_logger(args.save_dir, f'log_{split_name}')
        trainer = Trainer(args, log)
        checkpoint_path = os.path.join(args.save_dir, 'checkpoint')
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        #model = ParallelModel.from_pretrained(checkpoint_path, init_parallel=False)     # Should load linear weights from training
        model.to(args.device)
        eval_dataset, eval_dict = get_dataset(args, args.train_datasets, args.train_dir, tokenizer, split_name, test_datasets=args.eval_datasets, eval_dir=args.eval_dir)
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=1,
                                 sampler=SequentialSampler(eval_dataset))
        eval_preds, eval_scores = trainer.evaluate(model, eval_loader,
                                                   eval_dict, return_preds=True,
                                                   split=split_name)
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in eval_scores.items())
        log.info(f'Eval {results_str}')
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        log.info(f'Writing submission file to {sub_path}...')
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for uuid in sorted(eval_preds):
                csv_writer.writerow([uuid, eval_preds[uuid]])


if __name__ == '__main__':
    main()
