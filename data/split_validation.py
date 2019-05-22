
import sys
sys.path.append('../')
from utils.splitter import val_test_split

val_test_split('fewrel_val.json', 'fewrel_split_val.json', 'fewrel_split_test.json')