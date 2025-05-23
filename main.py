import argparse
import torch
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle CNN")
    parser.add_argument('--epochs', type=int, default=10, help="Nombre d'époques d'entraînement")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="Weight decay (L2 régularisation)")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                        help="Mode d'opération : train ou eval (default: train)")
    parser.add_argument('--cuda', action='store_true', help="Utiliser le GPU si disponible")
    parser.add_argument('--backend', type=str, choices=['pytorch', 'tensorflow'], required=True,
                        help="Choix du backend : pytorch ou tensorflow")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.backend == 'pytorch':
        from models.cnn_pytorch import get_pretrained_model
        from models.train_pytorch import Trainer
        from models.train_tensorflow import TrainerTF
        from utils import prep_pytorch as prep

        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        train_loader, test_loader = prep.get_data()

        model = get_pretrained_model().to(device)

        if args.mode == 'eval':
            model.load_state_dict(torch.load("Cheikh_Fall_model.torch"))

        trainer = Trainer(model, train_loader, test_loader, args.lr, args.wd, args.epochs, device)

        if args.mode == 'train':
            trainer.train(True,True)

        trainer.evaluate()

    elif args.backend == 'tensorflow':
        from models.cnn_tensorflow import get_pretrained_model
        from models.train_tensorflow import TrainerTF
        from utils import prep_tensorflow as prep

        train_loader, test_loader = prep.get_data()

        if args.mode == 'eval':
            model = tf.keras.models.load_model("Cheikh_Fall_model.tensorflow")
        else:
            model = get_pretrained_model()

        trainer = TrainerTF(model, train_loader, test_loader, args.lr, args.epochs)

        if args.mode == 'train':
            trainer.train(True, True)

        trainer.evaluate()


if __name__ == '__main__':
    main()
