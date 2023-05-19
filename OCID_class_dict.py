import numpy as np

cls_names = {
    'background'      : '0',
    'apple'           : '1',
    'ball'            : '2',
    'banana'          : '3',
    'bell_pepper'     : '4',
    'binder'          : '5',
    'bowl'            : '6',
    'cereal_box'      : '7',
    'mug'      : '8',
    'flashlight'      : '9',
    'food_bag'        : '10',
    'food_box'        : '11',
    'food_can'        : '12',
    'glue_stick'      : '13',
    'hand_towel'      : '14',
    'instant_noodles' : '15',
    'keyboard'        : '16',
    'kleenex'         : '17',
    'lemon'           : '18',
    'lime'            : '19',
    'marker'          : '20',
    'orange'          : '21',
    'peach'           : '22',
    'pear'            : '23',
    'potato'          : '24',
    'shampoo'         : '25',
    'soda_can'        : '26',
    'sponge'          : '27',
    'stapler'         : '28',
    'tomato'          : '29',
    'toothpaste'      : '30',
    'unknown'         : '31'
    }

colors = {
    '0': np.array([0, 0, 0]),
    '1': np.array([ 211, 47, 47 ]),
    '2': np.array([  0, 255, 0]),
    '3': np.array([123, 31, 162]),
    '4': np.array([ 81, 45, 168 ]),
    '5': np.array([ 48, 63, 159 ]),
    '6': np.array([25, 118, 210]),
    '7': np.array([ 2, 136, 209 ]),
    '8': np.array([  153, 51, 102 ]),
    '9': np.array([ 0, 121, 107 ]),
    '10': np.array([ 56, 142, 60 ]),
    '11': np.array([  104, 159, 56  ]),
    '12': np.array([ 175, 180, 43 ]),
    '13': np.array([  251, 192, 45  ]),
    '14': np.array([  255, 160, 0 ]),
    '15': np.array([ 245, 124, 0 ]),
    '16': np.array([ 230, 74, 25 ]),
    '17': np.array([  93, 64, 55 ]),
    '18': np.array([ 97, 97, 97 ]),
    '19': np.array([  84, 110, 122  ]),
    '20': np.array([ 255, 255, 102]),
    '21': np.array([ 0, 151, 167 ]),
    '22': np.array([  153, 255, 102  ]),
    '23': np.array([  51, 255, 102 ]),
    '24': np.array([  0, 255, 255  ]),
    '25': np.array([  255, 255, 255  ]),
    '26': np.array([  255, 204, 204  ]),
    '27': np.array([  153, 102, 0 ]),
    '28': np.array([  204, 255, 204  ]),
    '29': np.array([ 204, 255, 0  ]),
    '30': np.array([  255, 0, 255  ]),
    '31': np.array([ 194, 24, 91 ]),
}

colors_list = list(colors.values())
cls_list = list(cls_names.keys())