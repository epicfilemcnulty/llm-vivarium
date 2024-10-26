import torch
import yaml
import os
import argparse
import transformers
import bitsandbytes as bnb
from tqdm import tqdm
from transformers import TrainingArguments, Trainer
from transformers.optimization import get_cosine_schedule_with_warmup
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from bltzr import SqlDataModule, SqlDatasetConfig, SqlDataset, Tokenizer

class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits
 
        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
 
        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
 
        return lm_loss
 
    def save_model(self, output_dir, _internal_call=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def run(args):
    config = read_yaml_file(args.config_path)
    tokenizer = Tokenizer()
    if "base_model" in config:
        model_dir = config["base_model"]
        model = MambaLMHeadModel.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
    else:
        # Create base config dictionary with required parameters
        model_config_dict = {
            "d_model": config["d_model"],
            "n_layer": config["n_layer"],
            "vocab_size": len(tokenizer.vocab),
            "ssm_cfg": config["ssm_cfg"],
        }

        # Add attention parameters if they exist in config
        if "attn_layer_idx" in config:
            model_config_dict["attn_layer_idx"] = config["attn_layer_idx"]
        if "attn_cfg" in config:
            model_config_dict["attn_cfg"] = config["attn_cfg"]

        # Create MambaConfig with the assembled parameters
        model_config = MambaConfig(**model_config_dict)
        model = MambaLMHeadModel(config=model_config, device="cuda", dtype=torch.bfloat16)

    data_config = SqlDatasetConfig(
        db_host=args.host, db_pass=args.password, db_user=args.user, 
        db_name=args.database, dataset_table=config["dataset_table"], 
        window_size=config["chunk_size"], 
        with_metadata=bool(os.environ.get('LLM_TRAIN_WITH_METADATA')), 
        batch_size=args.batch)
    train_data = SqlDataModule(data_config)
    output_dir = "trainees/" + config["model_name"]

    if "optimizer" in config and config["optimizer"] == 'full':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))
    else:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=len(train_data.dataset))

    trainer = MambaTrainer(
        model=model,
        train_dataset=train_data.dataset,
        data_collator=train_data.data_collator,
        optimizers=(optimizer, lr_scheduler),
        args=TrainingArguments(
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            max_grad_norm = config["max_grad_norm"],
            max_steps = config["max_steps"],
            num_train_epochs=config["num_train_epochs"],
            save_steps = config["save_steps"],
            save_total_limit = config["save_total_limit"],
            logging_steps=config["logging_steps"],
            bf16 = True,
            tf32 = True,
            seed=args.seed,
            output_dir=output_dir,
        ),
    )
    trainer.train(resume_from_checkpoint=args.checkpoint)
    trainer.save_model(output_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, required=False)
    parser.add_argument("-s", "--seed", type=int, default=42, required=False)
    parser.add_argument("-b", "--batch", type=int, default=100, required=False)
    parser.add_argument('-d', '--database', required=False, default=os.environ.get('LLM_TRAIN_DB'), type=str, help="Database name")
    parser.add_argument('-u', '--user', required=False, default=os.environ.get('LLM_TRAIN_DB_USER'), type=str, help="Database user name")
    parser.add_argument('-p', '--password', required=False, default=os.environ.get('LLM_TRAIN_DB_PASS'), type=str, help="Database user password")
    parser.add_argument('--host', required=False, default=os.environ.get('LLM_TRAIN_DB_HOST'), type=str, help="Database host")
    args = parser.parse_args()
    run(args)
