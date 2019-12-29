import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from dataset import NeatLoader
import argparse
from model.neat_lstm import NeatLSTM
from model.neat_wpe import NeatWPE
from model.neat_wpeu import NeatWPEU
from model.neat_full import NeatFull
from model.neat_cnn import NeatCnn
from model.neat_cnn_wpe import NeatCnnWpe
from model.neat_cnn_fwpe import NeatCnnFwpe
from model.neat_cnn_wpeu import NeatCnnWpeu
from model.neat_cnn_full import NeatCnnFull
from model.neat_cnn_attn import NeatCnnAttn
from model.neat_cnn_ue import NeatCnnUE
from model.neat_attn import NeatAttn
from model.neat_ue import NeatUE
from model.config import NeatLstmConfig, NeatCnnConfig
from model.metrics import all_ir_metrics
import utils
import numpy as np
from constants import *


def evaluate(y_truth_buffer, y_prob_buffer, num_labels, vector=True, k=5):
    f1_results, precision_results, recall_results, accuracy_results = [], [], [], []
    for y_truth, y_prob in zip(y_truth_buffer, y_prob_buffer):
        if vector:
            y_truth = np.argwhere(y_truth.flatten() == 1).flatten()
            y_prob = np.argwhere(y_prob.flatten() > 0.5).flatten()
        precision, recall, f1 = all_ir_metrics(y_truth, y_prob)
        f1_results.append(f1)
        precision_results.append(precision)
        recall_results.append(recall)
    return {
        "f1": np.mean(f1_results),
        "precision": np.mean(precision_results),
        "recall": np.mean(recall_results)
    }


def train_fn(args, data_loader, model):
    tb_writer = SummaryWriter(os.path.join(args.log_dir, exp_name))
    num_labels = len(data_loader.side_effects)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    max_train_buffer_len = 100
    total_loss, logging_loss = 0.0, 0.0
    total_reg_loss, logging_reg_loss = 0.0, 0.0

    for epoch in trange(int(args.n_epochs), desc="Epoch"):
        y_train_truth_buffer, y_train_prob_buffer = [], []
        data_loader.set_mode("train")

        for step, batch in tqdm(enumerate(data_loader), desc="Training", total=len(data_loader)):
            model.train()
            x_post_word_idxs, x_user_idxs, x_user_clusters, y_truth = batch
            x_post_word_idxs = x_post_word_idxs.to(args.device)
            x_user_idxs = x_user_idxs.to(args.device)
            x_user_clusters = x_user_clusters.to(args.device)
            y_truth_gpu = y_truth.to(args.device)

            outputs = model(x_post_word_idxs, x_user_idxs, x_user_clusters)
            logits = outputs[0]
            loss = loss_fn(logits.unsqueeze(0), y_truth_gpu.unsqueeze(0).float())
            if args.alpha > 0:
                reg_loss = model.regularize_loss()
                loss = loss + args.alpha * reg_loss
            loss.backward()
            total_loss += loss.item()

            if len(y_train_prob_buffer) < max_train_buffer_len:
                y_train_prob = torch.sigmoid(logits).cpu().data
                y_train_prob_buffer.append(y_train_prob)
                y_train_truth_buffer.append(y_truth)

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.grad_accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if args.log_every > 0 and (global_step + 1) % args.log_every == 0:
                train_results = evaluate(y_train_truth_buffer, y_train_prob_buffer, num_labels)

                loss_per_step = (total_loss - logging_loss) / args.log_every
                reg_loss_per_step = (total_reg_loss - logging_reg_loss) / args.log_every
                logging_loss = total_loss
                logging_reg_loss = total_reg_loss

                train_results["loss"] = loss_per_step
                train_results["reg_loss"] = reg_loss_per_step

                print("Train eval at {}:".format(global_step))
                print(train_results)
                user_credibility_summary = model.update_writer(tb_writer, data_loader, global_step)
                print(user_credibility_summary)
                tb_writer.add_scalars('train', train_results, global_step)
                y_train_prob_buffer = []   # reset training truth buffer
                y_train_truth_buffer = []  # reset training prediction buffer

            if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0:
                val_results = validate_fn(args, data_loader, model, loss_fn)
                print("Validate eval at {}:".format(global_step))
                print(val_results)
                tb_writer.add_scalars('validate', val_results, global_step)
                data_loader.set_mode("train")  # set back data loading mode to train

            if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
                # Save model checkpoint
                utils.save_model_checkpoint(state=dict(
                    epoch=epoch,
                    state_dict=dict([(key, value.to("cpu")) for key, value in model.state_dict().items()]),
                    ),
                    is_best=False,
                    output_dir=args.output_dir,
                    exp_name=exp_name,
                    step=global_step
                )
            global_step += 1
    return global_step


def validate_fn(args, data_loader, model, loss_fn, mode="val"):
    data_loader.set_mode(mode)
    num_labels = len(data_loader.side_effects)
    y_val_truth_buffer, y_val_prob_buffer = [], []

    total_loss = 0.0
    total_reg_loss = 0.0
    global_steps = 0

    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader),
                            desc="{} Iteration".format("Validation" if mode == "val" else "Testing")):
        model.eval()
        x_post_word_idxs, x_user_idxs, x_user_clusters, y_truth = batch
        x_post_word_idxs = x_post_word_idxs.to(args.device)
        x_user_idxs = x_user_idxs.to(args.device)
        x_user_clusters = x_user_clusters.to(args.device)
        y_truth_gpu = y_truth.to(args.device)

        outputs = model(x_post_word_idxs, x_user_idxs, x_user_clusters)
        logits = outputs[0]

        loss = loss_fn(logits.unsqueeze(0), y_truth_gpu.unsqueeze(0).float())
        if args.alpha > 0:
            reg_loss = model.regularize_loss()
            loss = loss + args.alpha * reg_loss
        total_loss += loss.item()

        y_train_prob = torch.sigmoid(logits).cpu().data
        y_val_prob_buffer.append(y_train_prob)
        y_val_truth_buffer.append(y_truth)
        global_steps += 1

    val_results = evaluate(y_val_truth_buffer, y_val_prob_buffer, num_labels)
    val_results["loss"] = total_loss / global_steps
    val_results["reg_loss"] = total_reg_loss / global_steps
    return val_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=OUTPUT_DOC_DIR, type=str, help="Input data dir.")
    parser.add_argument("--cache_dir", default="data/pod/cache", type=str, help="Input cache dir.")
    parser.add_argument("--meta_dir", default="data/pod/meta", type=str, help="Input meta data dir.")
    parser.add_argument("--w2v_path", default="data/glove.6B.50d.txt", type=str, help="Input word embedding path")
    parser.add_argument("--ue_path", default="data/pod/meta/user_expertise.pickle", type=str, help="Input user expertise")
    parser.add_argument("--us_path", default="", type=str, help="Input user style")
    parser.add_argument("--ue_size", default=100, type=int, help="User expertise dim")
    parser.add_argument("--num_styles", default=17, type=int, help="Number of stylistic features")
    parser.add_argument("--log_dir", default="exp_logs", type=str, help="Path to tensorboard logging")
    parser.add_argument("--pretrained", default="", type=str, help="Path to pre-trained model")
    parser.add_argument("--output_dir", default="exp_ckpt", type=str, help="Trained model output dir")
    parser.add_argument("--size", default=1.0, type=float, help="Dataset size")
    parser.add_argument("--model_name", default="neat", type=str, help="Model name")
    parser.add_argument("--log_every", default=10000, type=int, help="Number of steps per very logging")
    parser.add_argument("--eval_every", default=20000, type=int, help="Number of steps per very validation")
    parser.add_argument("--save_every", default=20000, type=int, help="Number of steps per very model saving")
    parser.add_argument("--grad_accumulate_steps", default=16, type=int,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--n_epochs", default=50, type=int, help="Number of train epochs")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam's epsilon")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Maximum gradient norm")
    parser.add_argument("--alpha", default=0.0, type=float, help="Coefficient for regularization loss")
    parser.add_argument("--ue_coeff", default=0.5, type=float, help="Coefficient for user expertise")

    return parser.parse_args()


def load_model(model_name, user_embed_size, num_styles, device, data_loader, ue_coeff=0.0):
    if model_name == "neat_lstm":
        model = NeatLSTM(len(data_loader.vocab), len(data_loader.side_effects),
                         NeatLstmConfig.get_common())
    elif model_name == "neat_wpe":
        model = NeatWPE(len(data_loader.vocab), len(data_loader.users), len(data_loader.side_effects),
                        NeatLstmConfig.get_common())
    elif model_name == "neat_wpeu":
        model = NeatWPEU(len(data_loader.vocab), len(data_loader.users), user_embed_size, len(data_loader.side_effects),
                         NeatLstmConfig.get_common())
    elif model_name == "neat_ue":
        model = NeatUE(len(data_loader.vocab), len(data_loader.users), user_embed_size, len(data_loader.side_effects),
                       NeatLstmConfig.get_common())
    elif model_name == "neat_attn":
        model = NeatAttn(len(data_loader.vocab), len(data_loader.users), user_embed_size,
                         len(data_loader.clusters), len(data_loader.side_effects),
                         NeatLstmConfig.get_common())
    elif model_name == "neat_full":
        model = NeatFull(len(data_loader.vocab), len(data_loader.users), user_embed_size,
                         len(data_loader.clusters), len(data_loader.side_effects),
                         NeatLstmConfig.get_common())
    elif model_name == "neat_cnn":
        model = NeatCnn(len(data_loader.vocab), len(data_loader.side_effects), data_loader.max_post_len,
                        NeatCnnConfig.get_common())
    elif model_name == "neat_cnn_wpe":
        model = NeatCnnWpe(len(data_loader.vocab), len(data_loader.users),
                           len(data_loader.side_effects), data_loader.max_post_len,
                           NeatCnnConfig.get_common())
    elif model_name == "neat_cnn_fwpe":
        data_loader.thank_users = data_loader.thank_users.to(device)
        data_loader.thank_scores = data_loader.thank_scores.to(device)
        model = NeatCnnFwpe(len(data_loader.vocab), len(data_loader.users), num_styles,
                            len(data_loader.side_effects), data_loader.max_post_len,
                            data_loader.thank_users, data_loader.thank_scores,
                            NeatCnnConfig.get_common())
    elif model_name == "neat_cnn_wpeu":
        model = NeatCnnWpeu(len(data_loader.vocab), len(data_loader.users), user_embed_size,
                            len(data_loader.side_effects), data_loader.max_post_len,
                            NeatCnnConfig.get_common(ue_coeff=ue_coeff))
    elif model_name == "neat_cnn_ue":
        model = NeatCnnUE(len(data_loader.vocab), len(data_loader.users), user_embed_size,
                          len(data_loader.side_effects), data_loader.max_post_len,
                          NeatCnnConfig.get_common(ue_coeff=ue_coeff))
    elif model_name == "neat_cnn_full":
        model = NeatCnnFull(len(data_loader.vocab), len(data_loader.users), user_embed_size,
                            len(data_loader.clusters), len(data_loader.side_effects),
                            data_loader.max_post_len, NeatCnnConfig.get_common())
    elif model_name == "neat_cnn_attn":
        model = NeatCnnAttn(len(data_loader.vocab), len(data_loader.users), user_embed_size,
                            len(data_loader.clusters), len(data_loader.side_effects),
                            data_loader.max_post_len, NeatCnnConfig.get_common())
    else:
        raise ValueError("Unsupported model {}".format(model_name))
    return model


if __name__ == "__main__":
    program_args = parse_args()
    input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    program_args.device = input_device
    exp_name = utils.get_exp_name(program_args.model_name)
    program_args.exp_name = exp_name

    input_data_loader = NeatLoader(program_args.data_dir, program_args.meta_dir, program_args.cache_dir,
                                   program_args.size, program_args.w2v_path, program_args.ue_path, program_args.us_path)
    input_model = load_model(program_args.model_name, program_args.ue_size, program_args.num_styles,
                             input_device, input_data_loader, program_args.ue_coeff)
    if len(program_args.pretrained) > 0:
        loaded_state_dict = torch.load(program_args.pretrained)["state_dict"]
        input_model.load_state_dict(loaded_state_dict)
    else:
        input_model.init_weights(input_data_loader)
    input_model = input_model.to(input_device)
    train_fn(program_args, input_data_loader, input_model)
