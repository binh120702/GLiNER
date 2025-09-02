import argparse

from gliner import GLiNER
from gliner.evaluation import get_for_all_path, get_for_all_path_multiconer, get_for_all_path_vlsp
import os

def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="logs/model_12000", help="Path to model folder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to model folder")
    parser.add_argument('--data', type=str, default='data/ie_data/NER/', help='Path to the eval datasets directory')
    parser.add_argument('--eval_type', type=str, default='normal', help='Type of evaluation')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    model = GLiNER.from_pretrained(args.model, load_tokenizer=True).to("cuda:0")
    if args.eval_type == 'normal':
        get_for_all_path(model, -1, args.log_dir, args.data)
    elif args.eval_type == 'multiconer':
        get_for_all_path_multiconer(model, -1, args.log_dir, args.data)
    elif args.eval_type == 'vlsp':
        get_for_all_path_vlsp(model, -1, args.log_dir, args.data)