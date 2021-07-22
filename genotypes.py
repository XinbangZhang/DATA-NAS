from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# NASNet-A
NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)

AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

DARTS_V1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2

GDARTS = Genotype(
    normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),('sep_conv_5x5', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 2), ('sep_conv_3x3',3), ('sep_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1),
            ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))

DARTS_PLUS = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0),('sep_conv_5x5', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3',0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

SNAS = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),('dil_conv_3x3', 1), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect',0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=range(2, 6))

PDARTS = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0),
            ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

Gumbel_model_M4 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 3), ('skip_connect', 1), ('skip_connect', 2)],
            [('dil_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('dil_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 3)],
            [('skip_connect', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)]],
    reduce_concat=[2, 3, 4, 5])

Gumbel_model_M7 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('dil_conv_3x3', 3)],
            [('skip_connect', 1), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)],
            [('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)],
            [('skip_connect', 0), ('sep_conv_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

DARTS_gumbel = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1),
                                ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)],
                        normal_concat=[2, 3, 4, 5],
                        reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0),
                                ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1)],
                        reduce_concat=[2, 3, 4, 5])

gumbel_net_7i = Genotype(normal=[[('skip_connect', 0), ('sep_conv_3x3', 0)],
                                 [('skip_connect', 0), ('sep_conv_3x3', 1)],
                                 [('sep_conv_3x3', 1)],
                                 [('sep_conv_3x3', 1), ('skip_connect', 2)]], normal_concat=[2, 3, 4, 5],
                         reduce=[[('dil_conv_3x3', 0)],
                                 [('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
                                 [('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 3)],
                                 [('sep_conv_3x3', 0), ('skip_connect', 3), ('skip_connect', 4)]],
                         reduce_concat=[2, 3, 4, 5])

gumbel_net_4i = Genotype(normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
                                 [('sep_conv_3x3', 1), ('skip_connect', 0)],
                                 [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                                 [('sep_conv_3x3', 1), ('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
                         reduce=[[('skip_connect', 1), ('sep_conv_3x3', 1)],
                                 [('sep_conv_3x3', 2), ('skip_connect', 1)],
                                 [('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 3)],
                                 [('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)]],
                         reduce_concat=[2, 3, 4, 5])

gumbel_net_4i_mod = Genotype(normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
                                     [('sep_conv_3x3', 0), ('skip_connect', 2)],
                                     [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
                                     [('sep_conv_3x3', 0), ('skip_connect', 3)]], normal_concat=[2, 3, 4, 5],
                             reduce=[[('skip_connect', 1), ('sep_conv_3x3', 1)],
                                     [('sep_conv_3x3', 2), ('skip_connect', 1)],
                                     [('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 3)],
                                     [('sep_conv_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1)]],
                             reduce_concat=[2, 3, 4, 5])

gumbel_net_7_lgwd = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
                                     ('skip_connect', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1),
                                     ('max_pool_3x3', 0), ('sep_conv_3x3', 4)], normal_concat=[2, 3, 4, 5],
                             reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
                                     ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
                                     ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])
gumbel_net_4_lgwd = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2),
                                     ('skip_connect', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 2),
                                     ('max_pool_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],
                             reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
                                     ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3),
                                     ('max_pool_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])

gumel_net_no_4i_pretrain = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2),
                                            ('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 3),
                                            ('max_pool_3x3', 0), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5],
                                    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
                                            ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0),
                                            ('dil_conv_3x3', 2), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])

gumel_net_no_7i_pretrain = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2),
                                            ('max_pool_3x3', 0), ('skip_connect', 3), ('max_pool_3x3', 0),
                                            ('max_pool_3x3', 0), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5],
                                    reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0),
                                            ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
                                            ('dil_conv_5x5', 4), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

gumel_net_7i_pretrain_lgwd = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0),
                                              ('dil_conv_5x5', 2), ('skip_connect', 3), ('sep_conv_3x3', 1),
                                              ('skip_connect', 4), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5],
                                      reduce=[('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
                                              ('dil_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3),
                                              ('dil_conv_5x5', 3), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])

gumbel_net_1i_pretrain_lgwd = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 0),
                                               ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
                                               ('max_pool_3x3', 1), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5],
                                       reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0),
                                               ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 3),
                                               ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

gumbel_net_5i_pretrain_lgwd = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
                                               ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
                                               ('max_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=[2, 3, 4, 5],
                                       reduce=[('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
                                               ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 1),
                                               ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])

gumbel_net_2i_pretrain_lgwd = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0),
                                               ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1),
                                               ('max_pool_3x3', 3), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5],
                                       reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0),
                                               ('dil_conv_5x5', 1), ('skip_connect', 3), ('max_pool_3x3', 0),
                                               ('dil_conv_5x5', 3), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

gumbel_net_9i_pretrain_lgwd = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
                                               ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 3),
                                               ('max_pool_3x3', 0), ('max_pool_3x3', 3)], normal_concat=[2, 3, 4, 5],
                                       reduce=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 2),
                                               ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 1),
                                               ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)], reduce_concat=[2, 3, 4, 5])

gumbel_7_batchwise = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5],
    reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2),
            ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])

gumbel_7 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('skip_connect', 2),
            ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1),
            ('sep_conv_5x5', 0), ('skip_connect', 3), ('skip_connect', 4)], reduce_concat=[2, 3, 4, 5])

gumbel_7_batchwise_100 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2),
            ('skip_connect', 0), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])

gumbel_7_batchwise_50 = Genotype(
    normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0),
            ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 2),
            ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('max_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

gumbel_7_batchwise_30 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 3),
            ('max_pool_3x3', 1), ('dil_conv_5x5', 4), ('dil_conv_5x5', 2)], reduce_concat=[2, 3, 4, 5])

gumbel_7_batchwise_run2 = Genotype(
    normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0),
            ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5],
    reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('skip_connect', 2),
            ('max_pool_3x3', 1), ('dil_conv_5x5', 4), ('skip_connect', 2)], reduce_concat=[2, 3, 4, 5])

batchwise_3_3e4 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 2)],
            [('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 3)],
            [('avg_pool_3x3', 0), ('dil_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

batchwise_5_3e4 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('sep_conv_5x5', 3), ('skip_connect', 3)],
            [('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2)],
            [('dil_conv_3x3', 1), ('dil_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

visual_M4 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('sep_conv_5x5', 3), ('skip_connect', 3)],
            [('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 2)],
            [('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 3)]], reduce_concat=[2, 3, 4, 5])

visual_M4_mod = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 2)],
            [('skip_connect', 1), ('dil_conv_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0)],
            [('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 3)],
            [('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)]],
    reduce_concat=[2, 3, 4, 5])

batchwise_7_3e4 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 2)],
            [('skip_connect', 0), ('dil_conv_3x3', 3)],
            [('skip_connect', 1), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 2)],
            [('dil_conv_5x5', 1), ('sep_conv_5x5', 2), ('dil_conv_3x3', 2), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3)],
            [('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

batchwise_7_2e3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 1), ('skip_connect', 2)],
            [('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 3)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('skip_connect', 0)],
            [('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 1)],
            [('max_pool_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 3)]], reduce_concat=[2, 3, 4, 5])

batchwise_3_1e3 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('dil_conv_3x3', 2)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_3x3', 3)],
            [('sep_conv_5x5', 1), ('skip_connect', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 1), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 2)],
            [('skip_connect', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)],
            [('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4)]], reduce_concat=[2, 3, 4, 5])

batchwise_5_1e3 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 3), ('skip_connect', 1), ('skip_connect', 2)],
            [('dil_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('dil_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 3)],
            [('skip_connect', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)]],
    reduce_concat=[2, 3, 4, 5])

batchwise7_1e3 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)]], reduce_concat=[2, 3, 4, 5])

batchwise_7_3e4_mod1 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('dil_conv_3x3', 3)],
            [('skip_connect', 1), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)],
            [('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)],
            [('skip_connect', 0), ('sep_conv_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

visual_M7 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 1)],
            [('skip_connect', 0), ('dil_conv_3x3', 3), ('skip_connect', 2)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)],
            [('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)],
            [('skip_connect', 0), ('sep_conv_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

batchwise_7_3e4_mod2 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('dil_conv_3x3', 3)],
            [('skip_connect', 1), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)],
            [('dil_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 2), ('dil_conv_3x3', 3)],
            [('skip_connect', 0), ('sep_conv_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

test_geo = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 1)],
            [('sep_conv_5x5', 2)],
            [('skip_connect', 1), ('dil_conv_5x5', 3)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)],
            [('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 3)]],
    reduce_concat=[2, 3, 4, 5])

DARTS_top3 = Genotype(
    normal=[[('skip_connect', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 0)],
            [('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

DARTS_top4 = Genotype(
    normal=[[('skip_connect', 0), ('dil_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 3)],
            [('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 0), ('skip_connect', 3)]],
    reduce_concat=[2, 3, 4, 5])

epoch20 = Genotype(
    normal=[[('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 1)],
            [('dil_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 3)],
            [('sep_conv_3x3', 4), ('dil_conv_5x5', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('avg_pool_3x3', 0), ('dil_conv_3x3', 0)],
            [('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 1)],
            [('dil_conv_5x5', 0), ('skip_connect', 2)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3)]], reduce_concat=[2, 3, 4, 5])

epoch60 = Genotype(
    normal=[[('skip_connect', 0), ('dil_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 1)],
            [('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3)],
            [('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)]], reduce_concat=[2, 3, 4, 5])

epoch40 = Genotype(
    normal=[[('skip_connect', 0), ('dil_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 1)],
            [('max_pool_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 3)],
            [('dil_conv_5x5', 1), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)]], reduce_concat=[2, 3, 4, 5])

epoch80 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 2), ('sep_conv_5x5', 2)],
            [('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('dil_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 3), ('dil_conv_5x5', 3)],
            [('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

epoch100 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 2)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 2), ('sep_conv_3x3', 3)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('dil_conv_3x3', 1)],
            [('max_pool_3x3', 3), ('sep_conv_3x3', 4)]], reduce_concat=[2, 3, 4, 5])

camera_ready_4 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('skip_connect', 1), ('skip_connect', 2), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 3), ('skip_connect', 1), ('skip_connect', 2)],
            [('dil_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('dil_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 3)],
            [('skip_connect', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)]],
    reduce_concat=[2, 3, 4, 5])

camera_ready_7 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 3), ('skip_connect', 3)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 2), ('skip_connect', 2)],
            [('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3),
             ('skip_connect', 3)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)]],
    reduce_concat=[2, 3, 4, 5])

softmax_L1_1_M7_epoch50 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 4), ('sep_conv_3x3', 3), ('sep_conv_5x5', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)],
            [('skip_connect', 3), ('skip_connect', 4), ('dil_conv_3x3', 4)]], reduce_concat=[2, 3, 4, 5])

softmax_L1_1_M7_epoch100 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)],
            [('sep_conv_3x3', 3), ('skip_connect', 1), ('sep_conv_5x5', 3)],
            [('sep_conv_3x3', 3), ('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1)],
            [('avg_pool_3x3', 0), ('skip_connect', 3), ('dil_conv_3x3', 2), ('skip_connect', 2)],
            [('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 1)]],
    reduce_concat=[2, 3, 4, 5])

softmax_L1_1_M7_epoch130 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 0)],
            [('sep_conv_3x3', 3), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 4), ('sep_conv_3x3', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 0)],
            [('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('skip_connect', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 3), ('skip_connect', 3), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 4), ('dil_conv_5x5', 4)]],
    reduce_concat=[2, 3, 4, 5])

softmax_L1_1_M7_epoch150 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 2), ('skip_connect', 1), ('skip_connect', 0)],
            [('sep_conv_3x3', 3), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 2), ('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('dil_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 3), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2)],
            [('dil_conv_5x5', 2), ('skip_connect', 3), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3)]],
    reduce_concat=[2, 3, 4, 5])

OS_m7_300_1e3 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 2)]],
    normal_concat=range(2, 4),
    reduce=[[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('skip_connect', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0)]],
    reduce_concat=range(2, 4))

visual_OS = Genotype(
    normal=[[('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 2)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 3)]],
    normal_concat=range(2, 4),
    reduce=[[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('skip_connect', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 3), ('dil_conv_3x3', 0)],
            [('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)]],
    reduce_concat=range(2, 4))

visual_FS = Genotype(
    normal=[[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('dil_conv_5x5', 0), ('max_pool_3x3', 0)],
            [('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 2)]],
    normal_concat=range(2, 4),
    reduce=[[('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1)],
            [('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 2), ('max_pool_3x3', 3), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 1)],
            [('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0)]],
    reduce_concat=range(2, 4))

FS_m7_200 = Genotype(
    normal=[[('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 3), ('dil_conv_5x5', 0), ('max_pool_3x3', 0)]],
    normal_concat=range(2, 4),
    reduce=[[('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1)],
            [('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 2), ('max_pool_3x3', 3), ('dil_conv_5x5', 3), ('dil_conv_5x5', 2), ('skip_connect', 1)]],
    reduce_concat=range(2, 4))

# ab_search_epoch
# entrorelu-load-l1-05-150-gumbel-sample7-EXP-0.0003-20191110-120223
relu_l1_05_M7_epoch25 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)],
            [('sep_conv_3x3', 4), ('sep_conv_3x3', 3)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)],
            [('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 3), ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 4), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('skip_connect', 3)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M7_epoch50 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3)],
            [('sep_conv_3x3', 4), ('sep_conv_3x3', 3)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('skip_connect', 3), ('max_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_3x3', 1)],
            [('skip_connect', 4), ('skip_connect', 3), ('dil_conv_3x3', 4), ('skip_connect', 2)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M7_epoch100 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3)],
            [('sep_conv_3x3', 2), ('sep_conv_3x3', 4)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 3)],
            [('skip_connect', 4), ('dil_conv_3x3', 4), ('dil_conv_5x5', 3), ('skip_connect', 2)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M7_epoch150 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('sep_conv_3x3', 2), ('sep_conv_3x3', 3)],
            [('sep_conv_5x5', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
            [('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('skip_connect', 4), ('skip_connect', 3), ('dil_conv_5x5', 4)]],
    reduce_concat=[2, 3, 4, 5])

# entrorelu-l1-05-150-load-gumbel-sample4-EXP-0.0003-20191210-095642
relu_l1_05_M4_epoch50 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
            [('sep_conv_3x3', 1), ('skip_connect', 0)],
            [('skip_connect', 0)],
            [('skip_connect', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('dil_conv_5x5', 2), ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('dil_conv_5x5', 2), ('skip_connect', 3)],
            [('skip_connect', 2), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M4_epoch100 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 0)],
            [('skip_connect', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('skip_connect', 2), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2)],
            [('skip_connect', 3), ('avg_pool_3x3', 1), ('max_pool_3x3', 1)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M4_epoch150 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0)],
            [('skip_connect', 0)],
            [('skip_connect', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2)],
            [('skip_connect', 2), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)],
            [('skip_connect', 3), ('max_pool_3x3', 1), ('avg_pool_3x3', 1)]],
    reduce_concat=[2, 3, 4, 5])

# entrorelu-l1-05-150-gumbel-sample7-EXP-0.0003-20191211-111808
relu_l1_05_M7_random_epoch25 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 2)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)],
            [('dil_conv_3x3', 4), ('sep_conv_5x5', 4), ('sep_conv_3x3', 4)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M7_random_epoch50 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 4)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('sep_conv_5x5', 1)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M7_random_epoch100 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1), ('skip_connect', 0)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('skip_connect', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

relu_l1_05_M7_random_epoch150 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 2)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0)],
            [('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 3)],
            [('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)]],
    reduce_concat=[2, 3, 4, 5])

# entrorelu-l1-05-150-gumbel-sample4-EXP-0.0003-20191217-145429
relu_l1_05_M4_random_epoch50 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 0), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0)]], reduce_concat=[2, 3, 4, 5])

relu_l1_05_M4_random_epoch100 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 3)]], reduce_concat=[2, 3, 4, 5])

relu_l1_05_M4_random_epoch150 = Genotype(
    normal=[[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('avg_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# entrorelu-l1-05-300-gumbel-sample4-EXP-0.0003-20191216-175853
relu_l1_05_M4_long_random_epoch150 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('avg_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

relu_l1_05_M4_long_random_epoch200 = Genotype(
    normal=[[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1)],
            [('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2)]], reduce_concat=[2, 3, 4, 5])

relu_l1_05_M4_long_random_epoch250 = Genotype(
    normal=[[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
            [('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 2)],
            [('skip_connect', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 3)]], reduce_concat=[2, 3, 4, 5])

# ab-1.0-0.25-entrorelu-l1-0.5-150-sample7-0.0003-20191130-161048
relu_l1_05_M7_random_epoch150_ab_1_025 = Genotype(
    normal=[[('skip_connect', 0)], [('skip_connect', 1)],
            [('skip_connect', 0)], [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('skip_connect', 3), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 4), ('skip_connect', 3), ('skip_connect', 2)]], reduce_concat=[2, 3, 4, 5])

relu_l1_05_M7_random_epoch50_ab_1_025 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 1), ('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# ab-1.0-0.75-entrorelu-l1-0.5-150-sample7-0.0003-20191130-161018
relu_l1_05_M7_random_epoch150_ab_1_075 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('sep_conv_3x3', 3), ('skip_connect', 1)],
            [('skip_connect', 1), ('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 2)],
            [('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 3)],
            [('skip_connect', 2), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 3)]],
    reduce_concat=[2, 3, 4, 5])

# ab-1.0-0.8-entrorelu-l1-0.5-150-sample7-0.0003-20191201-101035
relu_l1_05_M7_random_epoch150_ab_1_08 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)],
            [('skip_connect', 0), ('sep_conv_5x5', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0)],
            [('sep_conv_5x5', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('dil_conv_5x5', 2)],
            [('dil_conv_5x5', 3), ('dil_conv_5x5', 4), ('sep_conv_5x5', 4), ('max_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# ab-0.2-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20191128-175241
relu_l1_05_M7_random_epoch150_ab_02_05 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1), ('skip_connect', 2)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 2), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 4), ('sep_conv_5x5', 4)]],
    reduce_concat=[2, 3, 4, 5])

# ab-0.3-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20191128-175306
relu_l1_05_M7_random_epoch150_ab_03_05 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('avg_pool_3x3', 0)],
            [('sep_conv_3x3', 2), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('skip_connect', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)],
            [('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 3), ('sep_conv_3x3', 3)]],
    reduce_concat=[2, 3, 4, 5])

# ab-2-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20191128-175202
relu_l1_05_M7_random_epoch150_ab_2_05 = Genotype(
    normal=[[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1), ('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('avg_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 0), ('sep_conv_3x3', 3)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 4)]], reduce_concat=[2, 3, 4, 5])

# ab-3-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20191128-175222
relu_l1_05_M7_random_epoch150_ab_3_05 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('skip_connect', 2)],
            [('max_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 3), ('dil_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 4), ('dil_conv_3x3', 4)]],
    reduce_concat=[2, 3, 4, 5])

# abmod-1-0.25-entrorelu-l1-0.5-150-sample7-0.0003-20191229-114147
relu_l1_05_M7_random_epoch150_abmod_1_025 = Genotype(
    normal=[[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2)],
            [('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 3)]],
    normal_concat=[2, 3, 4, 5], reduce=[
        [('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 0)],
        [('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)],
        [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 2), ('sep_conv_5x5', 0)],
        [('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 4), ('dil_conv_3x3', 2), ('max_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# abmod-1-0.75-entrorelu-l1-0.5-150-sample7-0.0003-20191229-114103
relu_l1_05_M7_random_epoch150_abmod_1_075 = genotype = Genotype(
    normal=[[('skip_connect', 0), ('dil_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 4), ('sep_conv_3x3', 3)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
            [('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('dil_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 1), ('skip_connect', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)],
            [('sep_conv_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-2-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200102-201407
relu_l1_05_M7_random_epoch150_abmodvar_2_05 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('skip_connect', 0)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1-0.75-entrorelu-l1-0.5-150-sample7-0.0003-20200102-201055
relu_l1_05_M7_random_epoch150_abmodvar_1_075 = Genotype(
    normal=[[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 0)],
            [('skip_connect', 1)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
            [('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1)],
            [('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 2), ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('skip_connect', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1-0.25-entrorelu-l1-0.5-150-sample7-0.0003-20200102-201139
relu_l1_05_M7_random_epoch150_abmodvar_1_025 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0)],
            [('skip_connect', 0)],
            [('skip_connect', 0)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('max_pool_3x3', 1)],
            [('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 3), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
            [('skip_connect', 4), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-0.2-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200102-201453
relu_l1_05_M7_random_epoch150_abmodvar_02_05 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 0)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 2), ('dil_conv_5x5', 2)],
            [('dil_conv_5x5', 2), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('skip_connect', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200107-160246
relu_l1_05_M7_random_epoch150_abmodvar_1_05 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('dil_conv_5x5', 2), ('skip_connect', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 0)],
            [('sep_conv_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-0.3-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200104-102436
relu_l1_05_M7_random_epoch150_abmodvar_03_05 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('skip_connect', 0)],
            [('skip_connect', 2), ('dil_conv_5x5', 4), ('skip_connect', 3), ('dil_conv_5x5', 2)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-3-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200104-102504
relu_l1_05_M7_random_epoch150_abmodvar_3_05 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1), ('avg_pool_3x3', 0)],
            [('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0)],
            [('dil_conv_3x3', 3), ('sep_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-0.3-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200105-214742
relu_l1_05_M7_random_epoch150_abmodvar_03_05_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0)],
            [('sep_conv_3xrelu_l1_05_M7_random_epoch150_abmodvar_03', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3),
             ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 3), ('dil_conv_5x5', 4)]],
    reduce_concat=[2, 3, 4, 5])
# val-abmod-3-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200114-100245
relu_l1_05_M7_random_epoch150_abmodvar_3_05_run2 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2)],
            [('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 0)]], reduce_concat=[2, 3, 4, 5]
)
# val-abmod-2-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200114-095733
relu_l1_05_M7_random_epoch150_abmodvar_2_05_run2 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0)], [('skip_connect', 1)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 0)],
            [('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 1)],
            [('skip_connect', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-0.2-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200116-221431
relu_l1_05_M7_random_epoch150_abmodvar_02_05_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 0)], [('skip_connect', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1)],
            [('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 2)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-0.3-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200116-221354
relu_l1_05_M7_random_epoch150_abmodvar_03_05_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0)], [('skip_connect', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('sep_conv_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_5x5', 4), ('max_pool_3x3', 1), ('sep_conv_5x5', 1)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1-0.75-entrorelu-l1-0.5-150-sample7-0.0003-20200120-095736
relu_l1_05_M7_random_epoch150_abmodvar_1_075_run2 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)],
            [('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 0)],
            [('dil_conv_5x5', 4), ('sep_conv_3x3', 4), ('sep_conv_5x5', 4), ('skip_connect', 3)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-0.2-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200119-150256
relu_l1_05_M7_random_epoch150_abmodvar_02_05_run3 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0)], [('skip_connect', 1)], [('skip_connect', 1)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('dil_conv_3x3', 1), ('skip_connect', 1), ('avg_pool_3x3', 0)],
            [('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 2), ('skip_connect', 2)],
            [('skip_connect', 2), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1-0.25-entrorelu-l1-0.5-150-sample7-0.0003-20200120-095558
relu_l1_05_M7_random_epoch150_abmodvar_1_025_run2 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0)], [('skip_connect', 0)], [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('skip_connect', 3), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 4), ('skip_connect', 3), ('skip_connect', 2)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.75-entrorelu-l1-0.5-150-sample4-0.0003-20200120-101828
relu_l1_05_M4_random_epoch150_abmodvar_1_075 = Genotype(
    normal=[[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('max_pool_3x3', 0)], 
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0)],
            [('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('dil_conv_5x5', 4), ('sep_conv_5x5', 3), ('skip_connect', 2)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.25-entrorelu-l1-0.5-150-sample4-0.0003-20200120-101748
relu_l1_05_M4_random_epoch150_abmodvar_1_025 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0)], [('skip_connect', 0)], [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0)],
            [('sep_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_3x3', 1)],
            [('skip_connect', 2), ('sep_conv_5x5', 1), ('avg_pool_3x3', 1)],
            [('skip_connect', 2), ('skip_connect', 3), ('sep_conv_5x5', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-0.2-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200120-164148
relu_l1_05_M4_random_epoch150_abmodvar_02_05 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0)], [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 0)],
            [('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-2.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200122-064255
relu_l1_05_M4_random_epoch150_abmodvar_2_05 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)],
            [('dil_conv_5x5', 4), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-3.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200122-123901
relu_l1_05_M4_random_epoch150_abmodvar_3_05 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 1)], [('skip_connect', 0), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 2)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3)],
            [('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.75-entrorelu-l1-0.5-150-sample4-0.0003-20200122-215834
relu_l1_05_M4_random_epoch150_abmodvar_1_075_run2 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 0)], [('skip_connect', 1), ('sep_conv_3x3', 2)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_5x5', 0), ('skip_connect', 3), ('sep_conv_5x5', 1)],
            [('avg_pool_3x3', 0), ('dil_conv_5x5', 4), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.25-entrorelu-l1-0.5-150-sample4-0.0003-20200122-170615
relu_l1_05_M4_random_epoch150_abmodvar_1_025_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0)], [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 3)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 3)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-2.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200124-003444
relu_l1_05_M4_random_epoch150_abmodvar_2_05_run2 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 1)], [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1), ('max_pool_3x3', 0)], [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-0.3-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200122-044230
relu_l1_05_M4_random_epoch150_abmodvar_03_05 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0)], [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 4)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-3.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200124-181652
relu_l1_05_M4_random_epoch150_abmodvar_3_05_run2 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], [('skip_connect', 1), ('skip_connect', 0)],
            [('sep_conv_3x3', 0), ('skip_connect', 1)], [('skip_connect', 1), ('sep_conv_3x3', 0)]],
    normal_concat=[2, 3, 4, 5], reduce=[[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)],
                                        [('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)],
                                        [('dil_conv_5x5', 2), ('max_pool_3x3', 1), ('skip_connect', 2)],
                                        [('sep_conv_5x5', 4), ('skip_connect', 2), ('sep_conv_5x5', 1)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-0.2-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200123-155251
relu_l1_05_M4_random_epoch150_abmodvar_02_05_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0)], [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.25-entrorelu-l1-0.5-150-sample4-0.0003-20200125-001321
relu_l1_05_M4_random_epoch150_abmodvar_1_025_run3 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0)], [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 0)],
            [('skip_connect', 2), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('skip_connect', 3), ('sep_conv_5x5', 0)],
            [('skip_connect', 3), ('dil_conv_5x5', 4), ('skip_connect', 2)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.75-entrorelu-l1-0.5-150-sample4-0.0003-20200125-170533
relu_l1_05_M4_random_epoch150_abmodvar_1_075_run3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 3)],
            [('sep_conv_5x5', 2), ('dil_conv_5x5', 4), ('max_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-0.3-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200123-222834
relu_l1_05_M4_random_epoch150_abmodvar_03_05_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0)], [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-2.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200126-172304
relu_l1_05_M4_random_epoch150_abmodvar_2_05_run3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0)], [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)], [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 3), ('dil_conv_5x5', 3), ('skip_connect', 2)],
            [('dil_conv_3x3', 4), ('sep_conv_3x3', 1), ('max_pool_3x3', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-3.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200127-082138
relu_l1_05_M4_random_epoch150_abmodvar_3_05_run3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 1)], [('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
            [('dil_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2)],
            [('sep_conv_5x5', 4), ('skip_connect', 3), ('avg_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-0.2-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200125-225407
relu_l1_05_M4_random_epoch150_abmodvar_02_05_run3 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0)], [('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2)],
            [('skip_connect', 2), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1)],
            [('dil_conv_5x5', 4), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-0.3-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200126-061649
relu_l1_05_M4_random_epoch150_abmodvar_03_05_run3 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1)], [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0)], [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 2)],
            [('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('skip_connect', 2)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.99-entrorelu-l1-0.5-150-sample4-0.0003-20200207-115935
relu_l1_05_M4_random_epoch150_abmodvar_1_099 = Genotype(
    normal=[[('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 1)],
            [('skip_connect', 1), ('avg_pool_3x3', 1)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 0)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('skip_connect', 1)],
            [('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)],
            [('sep_conv_5x5', 3), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 1)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.9-entrorelu-l1-0.5-150-sample4-0.0003-20200207-115854
relu_l1_05_M4_random_epoch150_abmodvar_1_09 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 0)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 0)],
            [('skip_connect', 3), ('skip_connect', 2), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('max_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.9-entrorelu-l1-0.5-150-sample7-0.0003-20200207-120015
relu_l1_05_M7_random_epoch150_abmodvar_1_09 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)],
            [('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 1)],
            [('sep_conv_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_3x3', 2)],
            [('dil_conv_5x5', 4), ('sep_conv_5x5', 4), ('sep_conv_3x3', 2), ('sep_conv_3x3', 4), ('dil_conv_3x3', 4)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.99-entrorelu-l1-0.5-150-sample7-0.0003-20200209-064635
relu_l1_05_M7_random_epoch150_abmodvar_1_099 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('avg_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0)],
            [('sep_conv_3x3', 0), ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)],
            [('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('avg_pool_3x3', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 0)],
            [('avg_pool_3x3', 2), ('avg_pool_3x3', 3), ('max_pool_3x3', 1), ('avg_pool_3x3', 1)],
            [('sep_conv_5x5', 0), ('sep_conv_3x3', 4), ('max_pool_3x3', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.99-entrorelu-l1-0.5-150-sample4-0.0003-20200209-113849
relu_l1_05_M4_random_epoch150_abmodvar_1_099_run2 = Genotype(
    normal=[[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 3)],
            [('max_pool_3x3', 1), ('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0)],
            [('dil_conv_3x3', 1), ('dil_conv_5x5', 1)],
            [('dil_conv_5x5', 4), ('sep_conv_5x5', 1), ('skip_connect', 3)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.9-entrorelu-l1-0.5-150-sample4-0.0003-20200209-094155
relu_l1_05_M4_random_epoch150_abmodvar_1_09_run2 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1)],
            [('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0)],
            [('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 2)],
            [('sep_conv_3x3', 3), ('dil_conv_5x5', 4), ('dil_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.99-entrorelu-l1-0.5-150-sample7-0.0003-20200213-230250
relu_l1_05_M7_random_epoch150_abmodvar_1_099_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('skip_connect', 0)],
            [('dil_conv_5x5', 2), ('skip_connect', 0)],
            [('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 2)],
            [('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 1)],
            [('skip_connect', 3), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4), ('sep_conv_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.9-entrorelu-l1-0.5-150-sample7-0.0003-20200213-230222
relu_l1_05_M7_random_epoch150_abmodvar_1_09_run2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 3)],
            [('skip_connect', 0), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 1)],
            [('sep_conv_3x3', 2), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 3), ('max_pool_3x3', 1)],
            [('sep_conv_5x5', 4), ('sep_conv_3x3', 3), ('sep_conv_5x5', 1), ('skip_connect', 3)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.99-entrorelu-l1-0.5-150-sample4-0.0003-20200211-225341
relu_l1_05_M4_random_epoch150_abmodvar_1_099_run3 = Genotype(
    normal=[[('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 0), ('skip_connect', 1)],
            [('avg_pool_3x3', 0), ('sep_conv_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 2)],
            [('dil_conv_3x3', 3), ('dil_conv_3x3', 2), ('avg_pool_3x3', 0)],
            [('sep_conv_3x3', 1), ('avg_pool_3x3', 4), ('sep_conv_5x5', 2)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.9-entrorelu-l1-0.5-150-sample4-0.0003-20200213-224737
relu_l1_05_M4_random_epoch150_abmodvar_1_09_run3 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 1), ('max_pool_3x3', 0)],
            [('sep_conv_3x3', 0), ('max_pool_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 0)],
            [('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)],
            [('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.99-entrorelu-l1-0.5-150-sample7-0.0003-20200211-230300
relu_l1_05_M7_random_epoch150_abmodvar_1_099_run3 = Genotype(
    normal=[[('skip_connect', 0), ('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('avg_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0)],
            [('avg_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 1)],
            [('skip_connect', 0), ('avg_pool_3x3', 0), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 1)],
            [('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
            [('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4), ('skip_connect', 3)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-1.0-0.9-entrorelu-l1-0.5-150-sample7-0.0003-20200211-225614
relu_l1_05_M7_random_epoch150_abmodvar_1_09_run3 = Genotype(
    normal=[[('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0)],
            [('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0)],
            [('max_pool_3x3', 1), ('avg_pool_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 2)],
            [('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2)],
            [('skip_connect', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-5.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200207-115026
relu_l1_05_M4_random_epoch150_abmodvar_5_05 = Genotype(
    normal=[[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_5x5', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 3), ('max_pool_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0)],
            [('skip_connect', 3), ('sep_conv_5x5', 4), ('avg_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-10.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200207-115105
relu_l1_05_M4_random_epoch150_abmodvar_10_05 = Genotype(
    normal=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('max_pool_3x3', 0)],
            [('sep_conv_5x5', 1), ('sep_conv_5x5', 2)],
            [('sep_conv_5x5', 0), ('max_pool_3x3', 0)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 3)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-15.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200207-115145
relu_l1_05_M4_random_epoch150_abmodvar_15_05 = Genotype(
    normal=[[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3)],
            [('sep_conv_3x3', 4), ('skip_connect', 2)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-5.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200208-210615
relu_l1_05_M7_random_epoch150_abmodvar_5_05 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_5x5', 1), ('skip_connect', 0), ('max_pool_3x3', 0), ('skip_connect', 1)],
            [('skip_connect', 1), ('skip_connect', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0)],
            [('skip_connect', 2), ('dil_conv_5x5', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 0)],
            [('skip_connect', 3), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-10.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200208-214058
relu_l1_05_M7_random_epoch150_abmodvar_10_05 = Genotype(
    normal=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)],
            [('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0)],
            [('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 1)],
            [('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 3), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-15.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200213-222309
relu_l1_05_M7_random_epoch150_abmodvar_15_05 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('skip_connect', 0)],
            [('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)],
            [('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1)],
            [('skip_connect', 2), ('skip_connect', 0), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 4)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-5.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200213-223954
relu_l1_05_M4_random_epoch150_abmodvar_5_05_run2 = Genotype(
    normal=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_5x5', 2)],
            [('sep_conv_3x3', 1), ('sep_conv_5x5', 2)],
            [('sep_conv_5x5', 2)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('max_pool_3x3', 1)],
            [('skip_connect', 2), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)],
            [('avg_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('dil_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-10.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200213-224034
relu_l1_05_M4_random_epoch150_abmodvar_10_05_run2 = Genotype(
    normal=[[('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('skip_connect', 1)],
            [('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0)],
            [('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('max_pool_3x3', 1)],
            [('sep_conv_5x5', 4), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-15.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200215-114109
relu_l1_05_M4_random_epoch150_abmodvar_15_05_run2 = Genotype(
    normal=[[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 1)],
            [('sep_conv_5x5', 1), ('skip_connect', 1), ('sep_conv_5x5', 0)],
            [('skip_connect', 1), ('sep_conv_5x5', 2)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-5.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200215-135401
relu_l1_05_M7_random_epoch150_abmodvar_5_05_run2 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('skip_connect', 1)],
            [('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('max_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 2)],
            [('skip_connect', 2), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 3)],
            [('sep_conv_5x5', 4), ('dil_conv_5x5', 4), ('sep_conv_3x3', 4)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-10.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200215-152008
relu_l1_05_M7_random_epoch150_abmodvar_10_05_run2 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('skip_connect', 0), ('max_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_3x3', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('dil_conv_5x5', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('skip_connect', 3)],
            [('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-15.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200217-024609
relu_l1_05_M7_random_epoch150_abmodvar_15_05_run2 = Genotype(
    normal=[[('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1)],
            [('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0)],
            [('sep_conv_5x5', 3), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 1)],
            [('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)],
            [('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('skip_connect', 2), ('sep_conv_5x5', 2)],
            [('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 3)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-5.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200213-230410
relu_l1_05_M4_random_epoch150_abmodvar_5_05_run3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('skip_connect', 1)],
            [('sep_conv_5x5', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)],
            [('sep_conv_5x5', 2), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0)],
            [('avg_pool_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 3)],
            [('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-10.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200215-123730
relu_l1_05_M4_random_epoch150_abmodvar_10_05_run3 = Genotype(
    normal=[[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('skip_connect', 1)],
            [('sep_conv_5x5', 0), ('max_pool_3x3', 0)],
            [('sep_conv_5x5', 0), ('max_pool_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('sep_conv_5x5', 3), ('max_pool_3x3', 0), ('avg_pool_3x3', 0)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 4)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-15.0-0.5-entrorelu-l1-0.5-150-sample4-0.0003-20200215-203148
relu_l1_05_M4_random_epoch150_abmodvar_15_05_run3 = Genotype(
    normal=[[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('skip_connect', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('sep_conv_5x5', 3)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('dil_conv_5x5', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 0)],
            [('sep_conv_5x5', 2), ('max_pool_3x3', 0), ('skip_connect', 2)],
            [('avg_pool_3x3', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 3)],
            [('skip_connect', 4), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# val-abmod-5.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200215-214329
relu_l1_05_M7_random_epoch150_abmodvar_5_05_run3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1)],
            [('sep_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('skip_connect', 0)],
            [('skip_connect', 0), ('dil_conv_5x5', 2), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 1)],
            [('skip_connect', 2), ('dil_conv_5x5', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('max_pool_3x3', 0)],
            [('dil_conv_3x3', 2), ('sep_conv_5x5', 3), ('dil_conv_5x5', 3), ('skip_connect', 2)],
            [('sep_conv_5x5', 4), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-10.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200215-234528
relu_l1_05_M7_random_epoch150_abmodvar_10_05_run3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 0)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 0)],
            [('sep_conv_5x5', 0), ('skip_connect', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_5x5', 0), ('sep_conv_5x5', 4), ('dil_conv_5x5', 3), ('skip_connect', 0)]],
    normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_5x5', 0), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0)],
            [('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 0)],
            [('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 2)],
            [('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# val-abmod-15.0-0.5-entrorelu-l1-0.5-150-sample7-0.0003-20200217-034728
relu_l1_05_M7_random_epoch150_abmodvar_15_05_run3 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)],
            [('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],
            [('max_pool_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 0)],
            [('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 0), ('skip_connect', 2)],
            [('sep_conv_5x5', 4), ('dil_conv_5x5', 4), ('skip_connect', 0), ('dil_conv_3x3', 1)]],
    reduce_concat=[2, 3, 4, 5])

# rebuttal_append_architecture
rebuttal_M4_visual1 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 2)],
            [('skip_connect', 1), ('dil_conv_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)],
            [('dil_conv_5x5', 4), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

rebuttal_M4_visual2 = Genotype(
    normal=[[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0)],
            [('skip_connect', 1), ('skip_connect', 2)],
            [('skip_connect', 1), ('sep_conv_3x3', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)],
            [('dil_conv_5x5', 4), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

rebuttal_M4_visual3 = Genotype(
    normal=[[('skip_connect', 0), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('sep_conv_3x3', 1), ('skip_connect', 0)],
            [('skip_connect', 0)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)],
            [('dil_conv_5x5', 4), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1)]], reduce_concat=[2, 3, 4, 5])

rebuttal_M7_visual1 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('skip_connect', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
            [('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 1)],
            [('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 1)],
            [('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2)],
            [('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)],
            [('skip_connect', 0), ('sep_conv_3x3', 0)]], reduce_concat=[2, 3, 4, 5])

# soft-entrorelu-l1-05-150-gumbel-sample7-EXP-0.0003-20201005-105219
soft_M7_e150_seed1 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],
            [('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0)],
            [('skip_connect', 2), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 0)],
            [('skip_connect', 2), ('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2)],
            [('skip_connect', 3), ('skip_connect', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 0)],
            [('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0)]],
    reduce_concat=[2, 3, 4, 5])

# soft-entrorelu-l1-05-150-gumbel-sample7-EXP-0.0003-20201005-105232
soft_M7_e150_seed2 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0)],
            [('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)],
            [('skip_connect', 1), ('skip_connect', 2)],
            [('skip_connect', 2), ('skip_connect', 1)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1)],
            [('dil_conv_5x5', 4), ('skip_connect', 2), ('skip_connect', 3), ('sep_conv_3x3', 0)]],
    reduce_concat=[2, 3, 4, 5])

# soft-entrorelu-l1-05-150-gumbel-sample4-EXP-0.0003-20201005-105031
soft_M4_e100_seed1 = Genotype(
    normal=[[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0)],
            [('skip_connect', 0), ('skip_connect', 1)],
            [('skip_connect', 0)], [('skip_connect', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 0)],
            [('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2)],
            [('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 1)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 4)]], reduce_concat=[2, 3, 4, 5])

# soft-entrorelu-l1-05-150-gumbel-sample4-EXP-0.0003-20201005-105132
soft_M4_e100_seed2 = Genotype(
    normal=[[('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)],
            [('skip_connect', 1), ('skip_connect', 0)],
            [('skip_connect', 1)], [('skip_connect', 4)]], normal_concat=[2, 3, 4, 5],
    reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)],
            [('max_pool_3x3', 0), ('avg_pool_3x3', 0)],
            [('max_pool_3x3', 0), ('dil_conv_5x5', 4)]], reduce_concat=[2, 3, 4, 5])