dataset = 'CULane'
data_root = '/root/autodl-tmp/CULane'  # Need to be modified before running
epoch = 30
batch_size = 4
optimizer = 'Adam'
learning_rate = 3.5e-4
weight_decay = 0.0001
momentum = 0.9
scheduler = 'cos'
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695
use_aux = False
griding_num = 200
backbone = 'mobilenetv3'
sim_loss_w = 0.0
shp_loss_w = 0.0
note = ''
log_path = '/root/autodl-tmp/culane_mobilenetv3'
finetune = None
resume = None
test_model = ''
test_work_dir = ''
tta = True
num_lanes = 4
var_loss_power = 2.0
auto_backup = True
num_row = 72
num_col = 81
train_width = 1600
train_height = 320
num_cell_row = 200
num_cell_col = 100
mean_loss_w = 0.05
fc_norm = True
crop_ratio = 0.6
