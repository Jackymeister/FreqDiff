# -*- coding: utf-8 -*-
import argparse
import os
import sys
import pickle
import logging
from datetime import datetime

import torch
import json
import numpy as np
from tqdm import tqdm
import random
import fitlog

# Compatibility for fitlog on NumPy >= 1.24 where np.str alias was removed.
if not hasattr(np, "str"):
    np.str = str


from torch.optim import *
import copy
import time

import scipy.sparse as sp
# import wandb

sys.path.append(os.path.dirname(__file__))
import utils_entity
from model import create_model_diffu, Att_Diffuse_model
from utils_entity import build_sub_graph
from knowledge_graph import _read_triplets_as_list

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")


class _NoopWandb:
    def finish(self):
        pass


wandb = _NoopWandb()


def history_seen_path(dataset, tim):
    filename = "h_r_seen_triple_{}.npy".format(str(tim))
    return os.path.join(DATA_DIR, dataset, "history_seq", filename)


def history_tail_path(dataset, tim):
    filename = "h_r_history_seq_{}.npz".format(str(tim))
    return os.path.join(DATA_DIR, dataset, "history_seq", filename)


def calculate_metrics(diffu_reps, ent_emb, true_triples):
    """
    计算 MRR、HIT@3 和 HIT@5 指标。

    参数:
    - diffu_reps: torch.Tensor, 形状为 [num_triples, embedding_dim]，表示尾实体的去噪分布。
    - ent_emb: torch.Tensor, 形状为 [num_entities, embedding_dim]，表示所有实体的嵌入。
    - true_triples: torch.Tensor, 形状为 [num_triples, 3]，表示真实的三元组 (s, r, o)。

    返回:
    - mrr: float，平均倒数排名。
    - hit3: float，HIT@3 指标。
    - hit5: float，HIT@5 指标。
    """
    num_triples = true_triples.size(0)
    ranks = []

    # 遍历每个三元组
    for i in range(num_triples):
        s, r, o = true_triples[i]
        s_emb = ent_emb[s]  # 头实体嵌入
        r_emb = ent_emb[r]  # 关系嵌入
        o_emb = ent_emb[o]  # 尾实体嵌入

        # 计算所有实体的得分
        scores = diffu_reps

        # 获取真实尾实体的得分
        true_score = scores[o]

        # 计算排名
        rank = (scores > true_score).sum().item() + 1
        ranks.append(rank)

    # 计算 MRR、HIT@3 和 HIT@5
    mrr = sum(1.0 / rank for rank in ranks) / num_triples
    hit3 = sum(1 for rank in ranks if rank <= 3) / num_triples
    hit5 = sum(1 for rank in ranks if rank <= 5) / num_triples

    return mrr, hit3, hit5

def test(args, model, model_name,
         history_times, query_times, all_ans_list,
         test_list,
         use_cuda,num_nodes, num_rels,mode='eval',ts=datetime.now().strftime("%Y-%m-%d") ,epoch=30,static_graph=None):
    """
    :param model: model used to test
    :param model_name: model state file name
    :param history_times: all time stamps in dataset
    :param query_times: all time stamps in testing dataset
    :param graph_dict: all graphs per day in dataset
    :param test_list: test triple snaps list
    :param all_ans_list: dict used for time-aware filtering (key and value are all int variable not tensor)
    :param head_ents: extremely frequent head entities causing popularity bias
    :param use_cuda: use cuda or cpu
    :param mode: 'eval' used in training process; or 'test' used for testing the best checkpoint
    :return: mrr for event object prediction
    """
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        logging.info("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint[
            'epoch']))  # use best stat checkpoint
        logging.info("\n" + "-" * 10 + "start testing " + model_name + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])

    rank_filter_list_sub, mrr_filter_list_sub = [], []
    rank_filter_list_obj, mrr_filter_list_obj = [], []
    rank_list_obj, mrr_list_obj = [], []
    tags_all_rel, tags_all_sub, tags_all_obj = [], [], []
    rank_filter_list_rel, mrr_filter_list_rel = [], []
    model.eval()
    with torch.no_grad():
        for time_idx, test_snap in enumerate(tqdm(test_list,colour='RED')):
            query_time = query_times[time_idx]
            query_idx = np.where(history_times == query_time)[0].item()
            input_time_list = history_times[query_idx - args.train_history_len: query_idx] # choose the time of train data 
            # history_glist = [graph_dict[tim] for tim in input_time_list] # choose train data with time

            # load test triplets: ( (s, r, o), ... ), len = all triplet in the same day
            test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
            output = test_triples_input.to(args.gpu)
                
            # output = torch.from_numpy(test_triples_input).long().cuda() if use_cuda else torch.from_numpy(test_triples_input).long()
            true_triples = getReverseSample(output,num_rels)
            seq_idx = true_triples[:, 0] * num_rels*2 + true_triples[:, 1]
            sequence = np.load(history_seen_path(args.dataset, query_time),
                               allow_pickle=True).tolist().toarray().reshape(true_triples.shape[0], args.his_max_len, 3) - 1
            sequence = torch.from_numpy(sequence).long().cuda() if use_cuda else torch.from_numpy(sequence).long()
            
            if args.add_static_graph:
                all_tail_seq = sp.load_npz(history_tail_path(args.dataset, query_time))  # 统计历史事实是否出现
                history_tail_seq = torch.Tensor(all_tail_seq[seq_idx.cpu()].todense())
                one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                if use_cuda:
                    history_tail_seq, one_hot_tail_seq = history_tail_seq.to(args.gpu), one_hot_tail_seq.to(args.gpu)
                seen_before_label = one_hot_tail_seq.gather(1,true_triples[:,2].unsqueeze(-1))
            else:
                history_tail_seq, one_hot_tail_seq,seen_before_label = None, None,None
            diffu_reps = []

            # true_triples = true_triples[:int(true_triples.shape[0] * 0.1)]

            for seed in range(1):
                scores, diffu_rep, weights, t, _,ent_emb = model(sequence,true_triples, False, use_cuda
                                                                 ,seen_before_label = seen_before_label,history_tail_seq=history_tail_seq,static_graph=static_graph)
                diffu_reps.append(diffu_rep) #-5,4  328 200

            scores_rec_diffu = model.diffu_rep_pre(diffu_reps,ent_emb,history_tail_seq, one_hot_tail_seq)
            # mrr, hit3, hit5 = calculate_metrics(scores_rec_diffu, ent_emb, true_triples)  #328 7000
            # print(f"MRR: {mrr:.4f}")
            # print(f"HIT@3: {hit3:.4f}")
            # print(f"HIT@5: {hit5:.4f}")


            write_pred_text(scores_rec_diffu, true_triples,(time_idx, time_idx),ts,args.dataset,epoch)
            
            mrr_filter_ent_obj, rank_filter_ent_obj , mrr, rank = utils_entity.get_total_rank(true_triples, scores_rec_diffu, all_ans_list[time_idx], eval_bz=1000, select_type=3,num_rel=num_rels,filer = args.filter)

            rank_filter_list_obj.append(copy.deepcopy(rank_filter_ent_obj))
            mrr_filter_list_obj.append(copy.deepcopy(mrr_filter_ent_obj))
            rank_list_obj.append(copy.deepcopy(rank))
            mrr_list_obj.append(copy.deepcopy(mrr))

    mrr_filter_all_obj, hits_filter_logs_obj = utils_entity.cal_ranks(rank_filter_list_obj, mode)
    mrr_all_obj, hits_logs_obj = utils_entity.cal_ranks(rank_list_obj, mode)
    return mrr_filter_all_obj,hits_filter_logs_obj,mrr_all_obj,hits_logs_obj

def write_pred_text(pred,ground,test_snap,time,d,epoch):
    file_path = os.path.join(RESULTS_DIR, "pred_result", d, "pred_{}.txt".format(time))
    # 确保路径存在，如果路径不存在则自动创建
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # print(file_path)
    with open(file_path,'a', encoding='utf-8') as file:
        file.writelines("\n---------{}-{}------------".format(epoch,test_snap))
        file.writelines("\n"+str(torch.argmax(pred[:50],-1).cpu().numpy().tolist()))
        file.writelines("\n"+str(ground[:,2][:50].cpu().numpy().tolist()))

def getReverseSample(test_triplets,num_rels):
    inverse_test_triplets = test_triplets[:, [2, 1, 0]] # inverse triplets 
    inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
    a_triples = torch.concat((test_triplets, inverse_test_triplets)) # [2,n_triplets,3]
    return a_triples

def annealing_schedule(cur_epoch,T0,TE):
    cur_tao = T0*(TE/T0)**(cur_epoch/100)
    return max(cur_tao,TE)
        
def run_experiment(args):
    # load graph data
    print("loading graph data")
    data = utils_entity.load_data(args.dataset)
    # train_list = utils_entity.split_by_time(
    #     data.train)  # [((s, r, o), ...), ...] len = num_date, do not contain inverse triplets
    # valid_list = utils_entity.split_by_time(data.valid)
    # test_list = utils_entity.split_by_time(data.test)
    # train_times = np.array(sorted(set(data.train[:, 3])))
    train_list = utils_entity.split_by_time(data.train)  # [((s, r, o), ...), ...] len = num_date, do not contain inverse triplets
    valid_list = utils_entity.split_by_time(data.valid)
    test_list = utils_entity.split_by_time(data.test)
    
    
    train_times = np.array(sorted(set(data.train[:, 3])))
    val_times = np.array(sorted(set(data.valid[:, 3])))
    test_times = np.array(sorted(set(data.test[:, 3])))
    
    # val_times = np.array(sorted(set(data.valid[:, 3])))
    # test_times = np.array(sorted(set(data.test[:, 3])))
    history_times = np.concatenate((train_times, val_times, test_times), axis=None)
    
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    print(num_nodes, num_rels)
    
    all_ans_list_test = utils_entity.load_all_answers_for_time_filter(data.test, num_rels, num_nodes)
    all_ans_list_valid = utils_entity.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    print(num_nodes, num_rels)
    seed =args.seed
    utils_entity.seed_everything(seed)
    

    # print("loading popularity bias data")
    # head_ents = json.load(open('./data/{}/head_ents.json'.format(args.dataset), 'r'))
    #

    # score_naming = '_' + args.score_aggregation
    encoder_naming = 'freq_v2_diffusion_mstep{}'.format(args.diffusion_steps)
    train_naming = '_t{}_lr{}_wd{}_m{}_r{}_e{}'.format(args.train_history_len, args.lr, args.wd,args.mask_rate,args.update_rel_rate,args.update_ent_rate)
    model_name = encoder_naming + train_naming

    result_dataset_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(result_dataset_dir, exist_ok=True)
    filename = os.path.join(result_dataset_dir, '{}{}.log'.format(model_name, args.alias))
    logging.basicConfig(level=logging.INFO, filename=filename)

    # models
    model_path = os.path.join(CHECKPOINT_DIR, "{}_fre_{}".format(args.dataset, args.fre))
    os.makedirs(model_path, exist_ok=True)
    model_state_file = os.path.join(model_path, model_name + args.alias + ".pt")
    legacy_model_state_file = os.path.join(model_path, model_name + args.alias)
    logging.info("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    
    fitlog.add_hyper_in_file(__file__)  # 记录本文件中写死的超参数

    diffu_rec = create_model_diffu(args)
    
    if args.add_static:
        static_triples = np.array(_read_triplets_as_list(os.path.join(DATA_DIR, args.dataset, "e-w-graph.txt"), load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes 
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    model = Att_Diffuse_model(diffu_rec, args,args.encoder,
                 num_nodes,
                 num_rels,
                 dropout=args.dropout,
                 max_time = len(history_times),
                 num_words=num_words,
                 num_static_rels=num_static_rels,
                 num_bases=args.n_bases)
    # create stat

    print(model)
    fitlog.add_other(model,"model")
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
        
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    per_epochs = (len(train_list)//args.accumulation_steps)

    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0= 50 * per_epochs,T_mult=2,eta_min=args.lr/100)



    # graph_dict = None
    # print("loading train, valid, test graphs...")
    # with open(os.path.join('./data/{}/'.format(args.dataset) , 'graph_dict.pkl'), 'rb') as fp:
    #     graph_dict = pickle.load(fp)

    if args.test and not os.path.exists(model_state_file):
        if os.path.exists(legacy_model_state_file):
            logging.info("Use legacy checkpoint path: {}".format(legacy_model_state_file))
            model_state_file = legacy_model_state_file
        else:
            print("--------------{} not exist, skip test----------------\n".format(model_state_file))
            return

    ts =  datetime.now().strftime("%Y-%m-%d")
    if args.test:
        print("----------------------------------------start testing----------------------------------------\n")
        test_filter_mrr_obj,test_filter_hit_obj,test_mrr_obj,test_hit_obj  = test(args,
             model=model,
             model_name=model_state_file,
             history_times=history_times,
             query_times=test_times,
             all_ans_list=all_ans_list_test,
             test_list=test_list,
             use_cuda=use_cuda,
             num_rels = num_rels,
             num_nodes= num_nodes,
             mode="test",
             ts= ts,
             static_graph=static_graph)
        # print(test_mrr_rel,test_mrr_obj,test_hit_rel,test_hit_obj)
        fitlog.add_metric({"dev":{"mrr":test_mrr_obj.item(),"hit_obj":test_hit_obj,"mrr_filter":test_filter_mrr_obj.item(),
                            "hit_filter":test_filter_hit_obj}
                                   ,"test":{"mrr":test_mrr_obj.item(),"hit_obj":test_hit_obj,"mrr_filter":test_filter_mrr_obj.item(),
                            "hit_filter":test_filter_hit_obj}},step=0,epoch=0)
        # wandb.log({"mrr":test_mrr_obj.item(),"hit_obj":test_hit_obj,"mrr_filter":test_filter_mrr_obj.item(),
        #                     "hit_filter":test_filter_hit_obj})
        
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_val_mrr, best_test_mrr = 0, 0
        best_epoch = 0
        accumulated = 0
        accumulation_steps = args.accumulation_steps
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            epoch_losses = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)  # shuffle based on time

            for batch_idx, train_sample_num in enumerate(tqdm(idx)):

                if train_sample_num == 0: continue  # make sure at least one history graph
                # train_list : [((s, r, o) on the same day)], len = uniq_t in train
                output = train_list[train_sample_num]  # all triplets in the next day to be predicted
                
                sequence = np.load(history_seen_path(args.dataset, train_times[train_sample_num]),
                                   allow_pickle=True).tolist().toarray().reshape(output.shape[0] * 2, args.his_max_len, 3) - 1
                    
                # generate history graph
                # history_glist = [graph_dict[tim] for tim in input_list]  # [(g), ...], len = valid history length
                output = torch.from_numpy(output).long().cuda() if use_cuda else torch.from_numpy(output).long()
                sequence = torch.from_numpy(sequence).long().cuda() if use_cuda else torch.from_numpy(sequence).long()
                true_triples = getReverseSample(output,num_rels)
                seq_idx = true_triples[:, 0] * num_rels * 2 + true_triples[:, 1]

                if args.add_static_graph:
                    all_tail_seq = sp.load_npz(history_tail_path(args.dataset, train_times[train_sample_num]))  # 统计历史事实是否出现
                    history_tail_seq = torch.Tensor(all_tail_seq[seq_idx.cpu()].todense())
                    one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                    if use_cuda:
                        history_tail_seq, one_hot_tail_seq = history_tail_seq.to(args.gpu), one_hot_tail_seq.to(args.gpu)
                    seen_before_label = one_hot_tail_seq.gather(1,true_triples[:,2].unsqueeze(-1))
                else:
                    history_tail_seq, one_hot_tail_seq,seen_before_label = None, None,None

                true_triples = true_triples[:int(true_triples.shape[0] * args.fre)]

                scores, diffu_rep, weights, t, mask_seq,emb_ent = model(sequence,true_triples, True, use_cuda,history_tail_seq,seen_before_label,static_graph)
                loss_diffu_value = model.loss_diffu_ce(diffu_rep, true_triples[:,2],history_tail_seq, one_hot_tail_seq,true_triples,mask_seq,emb_ent)
                loss = loss_diffu_value / accumulation_steps  # 将损失除以累积次数，这样使得每次累积的梯度相当于一个较大批次的梯度
                losses.append(loss.item())
                epoch_losses.append(loss.item())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(idx)-1:  # 检查累积次数是否达到阈值
                    fitlog.add_loss(np.mean(losses),name="Loss",step=batch_idx,epoch=epoch)
                    losses = []
                    optimizer.step()
                    scheduler.step()  #权重
                    optimizer.zero_grad()  # 重置梯度

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            print("Epoch {:04d}, AveLoss: {:.4f}, BestValMRR {:.4f}, BestTestMRR: {:.4f}, Model: {}, Dataset: {} "
                .format(epoch, epoch_loss, best_val_mrr, best_test_mrr, model_name, args.dataset))
            # validation and test
            if (epoch + 1) and (epoch + 1) % args.evaluate_every == 0:
                dev_filter_mrr_obj,dev_filter_hit_obj,dev_mrr_obj,dev_hit_obj = test(args,
                               model=model,
                               model_name=model_state_file,
                               history_times=history_times,
                               query_times=val_times,
                               all_ans_list=all_ans_list_valid,
                               test_list=valid_list,
                               num_rels = num_rels,
                               num_nodes = num_nodes,
                               use_cuda=use_cuda,
                               epoch = epoch,
                               mode="eval",
                               static_graph=static_graph)
                test_filter_mrr_obj,test_filter_hit_obj,test_mrr_obj,test_hit_obj  = test(args,
                                model=model,
                                model_name=model_state_file,
                                history_times=history_times,
                                query_times=test_times,
                                all_ans_list=all_ans_list_test,
                                test_list=test_list,
                                num_rels = num_rels,
                                num_nodes = num_nodes,
                                use_cuda=use_cuda,
                                mode="eval",
                                epoch=epoch,
                                ts=ts,
                                static_graph=static_graph)
                fitlog.add_metric({"dev":{"mrr":dev_mrr_obj.item(),"hit_obj":dev_hit_obj,"mrr_filter":dev_filter_mrr_obj.item(),
                            "hit_filter":dev_filter_hit_obj,"epoch":epoch}
                                   ,"test":{"mrr":test_mrr_obj.item(),"hit_obj":test_hit_obj,"mrr_filter":test_filter_mrr_obj.item(),
                            "hit_filter":test_filter_hit_obj,"epoch":epoch}},step=0,epoch=epoch)
                if dev_filter_mrr_obj < best_val_mrr:
                    accumulated += 1
                    if epoch >= args.n_epochs:
                        print("Max epoch reached! Training done.")
                        break
                    if accumulated >= args.patience:
                        print("Early stop triggered! Training done at epoch{}, best epoch is {}".format(epoch,
                                                                                                        best_epoch))
                        break
                else:
                    accumulated = 0
                    best_val_mrr = dev_filter_mrr_obj
                    best_test_mrr = test_filter_mrr_obj
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                    fitlog.add_best_metric({"dev":{"mrr":dev_mrr_obj.item(),"hit_obj":dev_hit_obj,"mrr_filter":dev_filter_mrr_obj.item(),
                            "hit_filter":dev_filter_hit_obj,"epoch":epoch}
                                   ,"test":{"mrr":test_mrr_obj.item(),"hit_obj":test_hit_obj,"mrr_filter":test_filter_mrr_obj.item(),
                            "hit_filter":test_filter_hit_obj,"epoch":epoch}})

        print('--- test best epoch model at epoch {}'.format(best_epoch))
        test(args,
             model=model,
             model_name=model_state_file,
             history_times=history_times,
             query_times=test_times,
             all_ans_list=all_ans_list_test,
             test_list=test_list,
             num_rels = num_rels,
             num_nodes = num_nodes,
             use_cuda=use_cuda,
             epoch = epoch,
             mode="test",
             ts=ts,
             static_graph=static_graph)
        fitlog.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FreqDiff')

    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--seed", type=int, default=2026,
                        help="random seed")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                        help="dataset: ICEWS14, ICEWS18, ICEWS05-15, or GDELT")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    
    # configuration for Diffusion Decoder
    parser.add_argument("--his_max_len", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--add_static",action='store_true', default=True)
    parser.add_argument("--no_static", action='store_false', dest='add_static',
                        help="disable the static entity-word graph")
    parser.add_argument("--add_static_graph",action='store_true', default=True)
    parser.add_argument("--no_static_graph", action='store_false', dest='add_static_graph',
                        help="disable the historical tail-frequency graph")
    parser.add_argument("--add_memory",action='store_true', default=False)
    parser.add_argument("--seen_addition",action='store_true', default=False)
    parser.add_argument("--add_ood_loss_energe",action='store_true', default=True)
    parser.add_argument("--kl_interst",action='store_true', default=False)
    parser.add_argument("--add_frequence",action='store_true', default=False)


    parser.add_argument("--temperature_object",type=float, default=0.5)
    parser.add_argument("--pattern_noise_radio",type=float, default=1)
    parser.add_argument("--encoder_params", type=dict, default=None)
    parser.add_argument("--concat_con", action='store_true', default=True,
                        help="perform layer normalization ")
    parser.add_argument("--refinements_radio", type=float, default=0,
                        help="perform layer normalization ")
    parser.add_argument("--heads", type=int, default=4,
                        help="perform layer normalization")
    
    parser.add_argument("--mask_rate", type=float, default=0)
    parser.add_argument("--update_rel_rate", type=float, default=0,
                        help="history length")
    parser.add_argument("--update_ent_rate", type=float, default=0,
                        help="history length")
    
    # configuration for gcn encoder
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder: rgcn/ compgcn")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n_hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--n_bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--self_loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer_norm", action='store_true', default=False,
                        help="perform layer normalization ") 
    parser.add_argument("--layer_norm_gcn", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")

    # configuration for sequences stat
    parser.add_argument("--train_history_len", type=int, default=3,
                        help="history length")
    # configuration for stat training
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--sample_nums", type=int, default=1,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience")
    parser.add_argument("--evaluate_every", type=int, default=1,
                        help="perform evaluation every n epochs")
    parser.add_argument("--alias", type=str, default='_freqdiff',
                        help="model naming alias, better start with _")
    
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="weight decay")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="weight decay")
    parser.add_argument("--grad_norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--accumulation_steps", type=int, default=2,
                        help="norm to clip gradient to")
    parser.add_argument("--filter", type=bool, default=True,
                        help="norm to clip gradient to")


    parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')

    parser.add_argument("--hidden_size", default=200, type=int, help="hidden size of model")
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='Dropout of embedding')
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of Transformer blocks')
    parser.add_argument('--num_blocks_cross', type=int, default=0, help='Number of Transformer blocks')
    parser.add_argument('--use_freq_enhance', action='store_true', default=True,
                        help='enable dual-stream frequency enhancement')
    parser.add_argument('--no_freq_enhance', action='store_false', dest='use_freq_enhance',
                        help='disable dual-stream frequency enhancement')
    parser.add_argument('--freq_dropout', type=float, default=0.1,
                        help='dropout used in the frequency stream')
    parser.add_argument('--group', type=int, default=4,
                        help='divisor used in FilterLayer: routing groups = hidden_size // group')
    parser.add_argument('--num_filters', type=int, default=8,
                        help='divisor used in FilterLayer: filter count = hidden_size // num_filters')
    parser.add_argument('--freq_gate_temperature', type=float, default=1.0,
                        help='temperature for context-aware dense frequency gate logits')
    parser.add_argument('--freq_context_pooling', type=str, default='mean',
                        choices=['mean', 'last'],
                        help='pooling strategy to build context-aware frequency weights from fused sequence representations')
    parser.add_argument('--freq_context_detach', action='store_true', default=False,
                        help='detach pooled context before frequency band-weight prediction for stability')
    parser.add_argument('--no_freq_context_detach', action='store_false', dest='freq_context_detach',
                        help='allow gradients through pooled context used by frequency band-weight prediction')
    parser.add_argument('--freq_exclude_query_token', action='store_true', default=True,
                        help='exclude the query/noisy tail slot from frequency modeling and fusion')
    parser.add_argument('--no_freq_exclude_query_token', action='store_false', dest='freq_exclude_query_token',
                        help='allow the query/noisy tail slot to enter frequency modeling')
    parser.add_argument('--freq_residual_alpha', type=float, default=0.1,
                        help='initial residual scale for frequency fusion')
    parser.add_argument('--learnable_freq_alpha', action='store_true', default=True,
                        help='learn residual scale for frequency fusion')
    parser.add_argument('--fixed_freq_alpha', action='store_false', dest='learnable_freq_alpha',
                        help='keep frequency residual scale fixed')
    parser.add_argument('--use_fft_loss', action='store_true', default=True,
                        help='enable auxiliary FFT-domain loss')
    parser.add_argument('--no_fft_loss', action='store_false', dest='use_fft_loss',
                        help='disable auxiliary FFT-domain loss')
    parser.add_argument('--fft_loss_weight', type=float, default=0.1,
                        help='weight for the auxiliary FFT-domain loss')
    parser.add_argument('--fft_loss_type', type=str, default='l1',
                        choices=['l1', 'l2', 'SmoothL1Loss', 'mix_loss'],
                        help='base loss used on real and imaginary FFT parts')
    parser.add_argument('--fft_loss_mode', type=str, default='sequence',
                        choices=['sequence', 'query_object'],
                        help='compute FFT auxiliary loss on sequence tokens or only query/target object token')
    parser.add_argument('--fft_loss_source', type=str, default='fused_rep',
                        choices=['input_rep', 'time_rep', 'freq_rep', 'fused_rep'],
                        help='which representation is matched to the clean sequence in FFT space')
    parser.add_argument('--fft_loss_detach_target', action='store_true', default=True,
                        help='detach the clean sequence target in FFT loss')
    parser.add_argument('--no_fft_loss_detach_target', action='store_false', dest='fft_loss_detach_target',
                        help='allow FFT loss gradients through the clean target sequence')
    parser.add_argument('--fft_loss_history_only', action='store_true', default=True,
                        help='compute FFT auxiliary loss only on history tokens, excluding query token')
    parser.add_argument('--no_fft_loss_history_only', action='store_false', dest='fft_loss_history_only',
                        help='compute FFT auxiliary loss on the full sequence including query token')

    parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
    parser.add_argument('--diffusion_steps', type=int, default=200, help='Diffusion step')
    parser.add_argument("--k_step", type=int, default=0, help="number of propagation rounds")
    parser.add_argument('--lambda_uncertainty', type=float, default=0.01, help='uncertainty weight')
    parser.add_argument('--noise_schedule', default='linear', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
    parser.add_argument('--rescale_timesteps', default=False, help='rescal timesteps')
    
    parser.add_argument('--scale', type=float, default=50, help='scale weight')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta_start weight')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta_end weight')
    parser.add_argument('--fre', type=float, default=1.0, help='')
    

    args = parser.parse_args()
    print(args)
    fitlog.set_log_dir(os.path.join(LOG_DIR, "data_{}".format(args.dataset)))  # 设定日志存储的目录
    fitlog.add_hyper(args)
    # 通过这种方式记录ArgumentParser的参数
    # wandb.init(config=args, project='FreqDiff',
    #         name='parameter',
    #         job_type="training",
    #         reinit=True)
    run_experiment(args)
    wandb.finish()
    sys.exit()
  
