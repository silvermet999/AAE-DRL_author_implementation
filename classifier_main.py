import argparse
import sys

import pandas as pd
import torch

import utils
from clfs import classifier


def parse_args(args):
    parser = argparse.ArgumentParser()

    # batch sizes and epochs
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--numEpochs', default=3, type=int)

    # when the discriminator loss reaches a threshold, we save the clf state dictionary
    parser.add_argument("--loss_threshold", default=0.9, type=float)


    # if test: ---train False
    parser.add_argument('--train', action='store_true')
    # unaug = unaugmented dataset = original dataset : if False then augmented dataset
    parser.add_argument("--unaug_dataset", action = "store_true")
    # Generate labels for synthetic dataset
    parser.add_argument("--label_gen", action="store_true")
    parser.add_argument("--synth_dataset", default=False) 
    parser.add_argument("--synth_dataset_path", default="/home/silver/PycharmProjects/AAEDRL/DRL/rl_bal1.csv") # path to unsupervised dataset
    # PLEASE USE THE ABSOLUTE PATH IF YOU GET A "NO FILE IS FOUND" ERROR!!!
    # save labels
    parser.add_argument("--labels_file", default = "labels.csv")
    # save classifier state dictionary
    parser.add_argument("--save_state_dict", default="best_model.pth")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    print("using", args.unaug_dataset)
    original_ds = utils.dataset(original=args.unaug_dataset, train=args.train)
    train_loader, val_loader = utils.dataset_function(original_ds, args.batch_size_train, args.batch_size_test, train=True)
    test_loader = utils.dataset_function(original_ds, args.batch_size_train, args.batch_size_test, train=False)
    if args.train:
        for epoch in range(args.numEpochs):
            # classifier pre-training
            train_loss = classifier.classifier_train(train_loader)
            # evaluation
            if epoch % 10 == 0:
                val_loss = classifier.classifier_val(val_loader)
                best_val_loss = args.loss_threshold
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(classifier.classifier.state_dict(), f'{args.save_state_dict}')

    else:
        # testing
        classifier.classifier_test(args.save_state_dict, test_loader)

    if args.label_gen:
        # generate labels
        print("gen labels")
        labels, cl = classifier.gen_labels(args.save_state_dict, args.synth_dataset_path, args.batch_size_train)
        labels = labels.detach().cpu().numpy()
        labels = pd.DataFrame(labels, columns=["attack_cat"])
        cl = pd.DataFrame(cl, columns=["confidence"])
        gen = pd.concat([labels, cl], axis=1)
        gen.to_csv(f"{args.labels_file}", index=False)


