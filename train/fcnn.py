"""
"""
import math

import torch.nn as nn
import torch.nn.init as init
#import custom_blocks
import torch.nn.functional as F

#__all__ = [
#    'FCNN', 'fcnn1', 'fcnn1_d', 'fcnn2', 'fcnn2_d', 'fcnn3', 'fcnn3_d', 'fcnn4', 'fcnn5'
#]


class FCNN(nn.Module):
    '''
    FCNN model
    '''
    def __init__(self, features, class_num=10):
        super(FCNN, self).__init__()
        self.features   = features

        num_blocks = len(self.features)
        output_dim = self.features[num_blocks-3].out_features
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, class_num)
        )
        #self.th         = custom_blocks.Thresholder()

        # Initialize weights and bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0.0)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        #a = self.th(x)
        a = x
        #x = self.smax(x)
        return F.log_softmax(x, dim=1), a


def make_layers(cfg, dropout=False, in_channels = 784):
    layers = []
    #in_channels = 784 #1024 #784
    num_linear = len(cfg)

    i = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Linear(in_channels, v)
            i += 1

            if dropout:
                p_d = 0
                if (i > num_linear-1):
                    p_d = 0.3*i/num_linear

                layers += [conv, nn.ReLU(inplace=True), nn.Dropout(p=p_d)]
            else:
                layers += [conv, nn.ReLU(inplace=True), nn.Dropout(p=0)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A' : [10],
    'Aa': [30],
    'Ab': [15],
    'Ac': [200],
    'Ad': [500],
    'Ae': [3000],
    'Af': [10000],
    'B' : [25, 25],
    'Ba': [50, 50],
    'Bb': [100, 100],
    'Bc': [200, 200],
    'Bd': [400, 400],
    'Be': [800, 800],
    'Bf': [1600, 1600],
    'Bg': [900, 500],
    'Bh': [1000, 1000],
    'C' : [25, 25, 25],
    'Ca': [50, 50, 50],
    'Cb': [100, 100, 100],
    'Cc': [200, 200, 200],
    'D' : [25 , 25 , 25 , 25],
    'Da': [50 , 50 , 50, 50],
    'Db': [100, 100, 100, 100],
    'Dc': [5 , 5 , 5 , 10],
    'Dd': [5 , 10, 5 , 10],
    'De': [5 , 10, 10, 10],
    'Df': [10, 10, 10, 10],
    'Dg': [5 , 7 , 10, 10],
    'Dh': [10, 5 , 10, 10],
    'Di': [7 , 7 , 10, 10],
    'Dj': [8 , 8 , 10, 10],
    'Dk': [12, 12, 12, 12],
    'Dl': [10, 9,   8,  7],
    'Dm': [20, 20, 20, 20],
    'Dn': [30, 30, 30, 30],
    'E' : [25 , 25 , 25 , 25 , 25],
    'Ea': [50 , 50 , 50 , 50, 50],
    'Eb': [100, 100, 100, 100, 100],
    'Ec': [10, 10, 5 , 5 , 5],
    'Ed': [100, 100, 100, 100, 100],
    'F' : [5 , 5 , 5 , 5 , 5 , 5 ],
    'Fa': [5 , 5 , 5 , 10, 10, 10],
    'Fb': [8 , 8 , 8 , 8 , 8 , 8 ],
    'Z':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

__all__ = [
    'fcnn_prune',
    'fcnn'  ,
    'fcnn1' , 'fcnn1_d' , 'fcnn1a', 'fcnn1a_d', 'fcnn1b', 'fcnn1b_d', 'fcnn1c_d', 'fcnn1d_d', 'fcnn1e_d', 'fcnn1f_d',
    'fcnn2' , 'fcnn2_d' , 'fcnn2a', 'fcnn2a_d', 'fcnn2b', 'fcnn2b_d', 'fcnn2c'  , 'fcnn2c_d', 'fcnn2d_d', 'fcnn2e', 'fcnn2e_d', 'fcnn2f', 'fcnn2g', 'fcnn2h',
    'fcnn3' , 'fcnn3_d' , 'fcnn3a', 'fcnn3a_d', 'fcnn3b', 'fcnn3b_d', 'fcnn3c'
    'fcnn4' , 'fcnn4_d' , 'fcnn4a', 'fcnn4a_d', 'fcnn4b', 'fcnn4b_d', 'fcnn4c', 'fcnn4c_d', 'fcnn4d', 'fcnn4d_d',
    'fcnn4e', 'fcnn4e_d', 'fcnn4f', 'fcnn4f_d', 'fcnn4g_d', 'fcnn4h_d',  'fcnn4i_d', 'fcnn4j_d', 'fcnn4l_d', 'fcnn4m_d', 'fcnn4n_d'
    'fcnn5' , 'fcnn5_d' , 'fcnn5a', 'fcnn5a_d', 'fcnn5b', 'fcnn5b_d', 'fcnn5c', 'fcnn5c_d', 'fcnn5d',
    'fcnn6' , 'fcnn6_d' , 'fcnn6a', 'fcnn6a_d', 'fcnn6b', 'fcnn6b_d'
]

def fcnn_prune(cfg, input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg, in_channels = input_dim), class_num = class_num)

def fcnn1(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['A'], in_channels = input_dim), class_num = class_num)

def fcnn1_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['A'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn1a(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Aa'], in_channels = input_dim), class_num = class_num)

def fcnn1a_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Aa'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn1b(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ab'], in_channels = input_dim), class_num = class_num)

def fcnn1b_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ab'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn1c_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ac'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn1d_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ad'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn1e_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ae'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn1f_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Af'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn2(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['B'], in_channels = input_dim), class_num = class_num)

def fcnn2_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['B'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn2a(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ba'], in_channels = input_dim), class_num = class_num)

def fcnn2a_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ba'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn2b(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bb'], in_channels = input_dim), class_num = class_num)

def fcnn2b_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bb'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn2c(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bc'], in_channels = input_dim), class_num = class_num)

def fcnn2c_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bc'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn2d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bd'], dropout=False, in_channels = input_dim), class_num = class_num)

def fcnn2d_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bd'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn2e(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Be'], dropout=False, in_channels = input_dim), class_num = class_num)

def fcnn2e_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Be'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn2f(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bf'], dropout=False, in_channels = input_dim), class_num = class_num)

def fcnn2g(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bg'], dropout=False, in_channels = input_dim), class_num = class_num)

def fcnn2h(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Bh'], dropout=False, in_channels = input_dim), class_num = class_num)

def fcnn3(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['C'], in_channels = input_dim), class_num = class_num)

def fcnn3_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['C'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn3a(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ca'], in_channels = input_dim), class_num = class_num)

def fcnn3a_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ca'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn3b(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Cb'], in_channels = input_dim), class_num = class_num)

def fcnn3b_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Cb'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn3c(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Cc'], in_channels = input_dim), class_num = class_num)

def fcnn4(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['D'], in_channels = input_dim), class_num = class_num)

def fcnn4_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['D'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4a(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Da'], in_channels = input_dim), class_num = class_num)

def fcnn4a_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Da'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4b(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Db'], in_channels = input_dim), class_num = class_num)

def fcnn4b_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Db'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4c(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dc'], in_channels = input_dim), class_num = class_num)

def fcnn4c_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dc'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dd'], in_channels = input_dim), class_num = class_num)

def fcnn4d_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dd'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4e(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['De'], in_channels = input_dim), class_num = class_num)

def fcnn4e_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['De'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4f(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Df'], in_channels = input_dim), class_num = class_num)

def fcnn4f_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Df'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4g_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dg'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4h_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dh'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4i_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Di'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4j_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dj'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4l_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dl'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4m_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dm'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn4n_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Dn'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn5(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['E'], in_channels = input_dim), class_num = class_num)

def fcnn5_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['E'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn5a(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ea'], in_channels = input_dim), class_num = class_num)

def fcnn5a_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ea'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn5b(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Eb'], in_channels = input_dim), class_num = class_num)

def fcnn5b_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Eb'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn5c(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ec'], in_channels = input_dim), class_num = class_num)

def fcnn5c_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ec'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn5d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Ed'], in_channels = input_dim), class_num = class_num)

def fcnn6(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['F'], in_channels = input_dim), class_num = class_num)

def fcnn6_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['F'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn6a(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Fa'], in_channels = input_dim), class_num = class_num)

def fcnn6a_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Fa'], dropout=True, in_channels = input_dim), class_num = class_num)

def fcnn6b(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Fb'], in_channels = input_dim), class_num = class_num)

def fcnn6b_d(input_dim = 784, class_num = 10):
    return FCNN(make_layers(cfg['Fb'], dropout=True, in_channels = input_dim), class_num = class_num)
