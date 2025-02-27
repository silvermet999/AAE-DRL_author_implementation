import argparse
import sys

import torch

import utils
from AAE import AAE_training, AAE_testing


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--numEpochs', default=101, type=int)
    # when the discriminator loss reaches a threshold, we save the AAE state dictionary
    parser.add_argument("--loss_threshold", default=0.6, type=float)
    # define number of interpolations and sample size per interpolation
    # !!!DESIRED DATASIZE = number of interpolations * sample size per interpolation!!!
    parser.add_argument("--n_inter", default=5, type=int) # we set it to 4 when --unaug_dataset = False
    parser.add_argument("--n_samples_per_inter", default=27321, type=int) # we set it to 43313 when --unaug_dataset = False


    # if test: ---train False
    parser.add_argument('--train', action='store_true')
    # unaug = unaugmented dataset = original dataset : if False then augmented dataset
    parser.add_argument("--unaug_dataset", action="store_true")
    parser.add_argument("--dataset_file", default="ds.csv")
    # PLEASE USE THE ABSOLUTE PATH IF YOU GET A NO FILE IS FOUND!!!
    # Save AAE state dictionary
    parser.add_argument("--save_state_dict", default="/home/silver/PycharmProjects/AAEDRL/AAE/aae3.pth")
    # Path to augmented dataset
    parser.add_argument('--X_ds', default="/home/silver/PycharmProjects/AAEDRL/DRL/rl_bal1.csv")
    # Path to augmented dataset's labels
    parser.add_argument('--y_ds', default="/home/silver/PycharmProjects/AAEDRL/clfs/labels.csv")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    print("using", args.unaug_dataset)
    dataset = utils.dataset(original=args.unaug_dataset, train=args.train)

    if args.train:
        train_loader, val_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                                          batch_size_o=args.batch_size_test, train=True)
        best_d_val_loss = args.loss_threshold
        for epoch in range(args.numEpochs):
            # Train AAE
            g_loss, d_loss = AAE_training.train_model(train_loader)
            print(f"Epoch {epoch + 1}/{args.numEpochs}, g loss: {g_loss}, d loss: {d_loss}")
            # Evaluate AAE
            if epoch % 10 == 0:
                g_val, d_val = AAE_training.evaluate_model(val_loader)
                print(f"g loss: {g_val}, d loss: {d_val}")
                # Save state dictionary
                if d_val < best_d_val_loss:
                    best_d_val_loss = d_val
                    torch.save({'epoch': epoch,
                                'enc_gen': AAE_training.encoder_generator.state_dict(),
                                'dec': AAE_training.decoder.state_dict(),
                                "disc": AAE_training.discriminator.state_dict(),
                                'val_loss': d_loss}, f"{args.save_state_dict}")

        # Generate samples and save
        d, c, b = AAE_training.sample_runs(args.n_inter, args.n_samples_per_inter)
        AAE_training.save_features_to_csv(d, c, b, args.dataset_file)

    else:
        # Test AAE
        test_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                             batch_size_o=args.batch_size_test, train=False)
        g_loss, d_loss = AAE_testing.test_model(test_loader, args.save_state_dict)
        print(f"g_loss: {g_loss}, d_loss: {d_loss}")

