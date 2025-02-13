import argparse


def parser_handle():
    '''
    input hyper parameters
    '''
    parser = argparse.ArgumentParser(description='REFINE')

    parser.add_argument('--gpu_id', type=str, default='0')

    parser.add_argument('--model', type=str, default='ResNet18',
                        help='the fixed pre-trained model')
    
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='the reprogramming dataset')
            
    parser.add_argument('--attack', type=str, default='BadNets',
                        help='the type of poisoning')
    
    parser.add_argument('--tlabel', type=int, default=0,
                        help='the label of poison')
    
    parser.add_argument('--refine_res', type=str, default='refine_res',
                        help='the directory of saving training results')
            
    parser.add_argument('--lmd', type=float, default=0, help='the coefficient for a weight norm penalty, to reduce overfitting')
    parser.add_argument('--mse', type=float, default=0.03, help='the coefficient for mse_loss')
    parser.add_argument('--sup', type=float, default=0.1, help='the coefficient for supcontract_loss')
    parser.add_argument('--lr', type=float, default=0.01, help='the learning rate')
    parser.add_argument('--optim', type=str, default='SGD', help='the type of optimization')
    parser.add_argument('--decay', type=float, default=0.8, help='the decay of learning rate')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=150, help='the max epoch for training the reprogramming model')
    parser.add_argument('--batch_size', type=int, default=256)

    return parser