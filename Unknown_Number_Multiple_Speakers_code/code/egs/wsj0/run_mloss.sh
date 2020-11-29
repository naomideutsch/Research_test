#!/bin/bash

# -- START IMPORTANT
# * If you have mixture wsj0 audio, modify `data` to your path that including tr, cv and tt.
# * If you just have origin sphere format wsj0 , modify `wsj0_origin` to your path and
# modify `wsj0_wav` to path that put output wav format wsj0, then read and run stage 1 part.
# After that, modify `data` and run from stage 2.

# wsj0_origin=/private/home/eliyan/graph_nn/speech_separation/csr_1/
# wsj0_wav=/private/home/eliyan/graph_nn/speech_separation/WSJ0/

#data="/Users/naomi/Documents/University/SecondDegree/SecondYear/Research/codes/Research_test/Unknown_Number_Multiple_Speakers_code/code/egs/wsj0/data_whamr"
data="/content/Research_test/Unknown_Number_Multiple_Speakers_code/code/egs/wsj0/data_whamr"

stage=2  # Modify this to control to start from which stage
# -- END

dumpdir=data  # directory to put generated json file
dumpdir=data_wham  # directory to put generated json file


# -- START Conv-TasNet Config
train_dir=$dumpdir/tr
valid_dir=$dumpdir/cv
evaluate_dir=$dumpdir/tt
separate_dir=$dumpdir/tt
sample_rate=8000
segment=4  # seconds
cv_maxlen=6  # seconds
# Network config
N=128
L=8
B=128
H=512
P=3
X=25
R=4
norm_type=gLN
causal=0
mask_nonlinear='relu'
C=2
# Training config
use_cuda=1
id='0'
ide=1 # VERY IMPORTANT!!!
epochs=1000000
half_lr=1
early_stop=0
max_norm=5
# minibatch
shuffle=1
batch_size=2
num_workers=4
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=0
# save and visualize
checkpoint=1
#continue_from="/Users/naomi/Documents/University/SecondDegree/SecondYear/Research/codes/Research_test/Unknown_Number_Multiple_Speakers_code/code/egs/wsj0/exp/train_wham_555__r8000_N128_L8_B128_H512_P3_X25_R4_C2_gLN_causal0_relu_epoch1000000_half1_norm5_bs2_worker4_adam_lr5e-4_mmt0_rnn_b_layer8_segment4_l20_lw1.0_tflip1_loss_every1_lr_decay0.92_tr/final.pth.tar"
continue_from="/content/drive/My Drive/Colab Notebooks/Research_Test/exp/train_wham_555__r8000_N128_L8_B128_H512_P3_X25_R4_C2_gLN_causal0_relu_epoch1000000_half1_norm5_bs2_worker4_adam_lr5e-4_mmt0_rnn_b_layer8_segment4_l20_lw1.0_tflip1_loss_every1_lr_decay0.92_tr/final.pth.tar"

print_freq=10
visdom=0
visdom_epoch=0
visdom_id="Conv-TasNet Training"
# evaluate
ev_use_cuda=1
cal_sdr=1
batch_size_eval=2
eval_every=1
# -- END Conv-TasNet Config

# DPRNN
lr_decay=0.92
rnn_hidden_dim=128
rnn_b_layer=8
lw=1.0
tflip=1
loss_every=1

# exp tag
tag="icml_submisson" # tag for managing experiments.


ngpu=1  # always 1

. ./utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh


if [ $stage -le 0 ]; then
  echo "Stage 0: Convert sphere format to wav format and generate mixture"
  local/data_prepare.sh --data ${wsj0_origin} --wav_dir ${wsj0_wav}

  echo "NOTE: You should generate mixture by yourself now.
You can use tools/create-speaker-mixtures.zip which is download from
http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip
If you don't have Matlab and want to use Octave, I suggest to replace
all mkdir(...) in create_wav_2speakers.m with system(['mkdir -p '...])
due to mkdir in Octave can not work in 'mkdir -p' way.
e.g.:
mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);
->
system(['mkdir -p ' output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);"
  exit 1
fi


if [ $stage -le 1 ]; then
  echo "Stage 1: Generating json files including wav path and duration"
  [ ! -d $dumpdir ] && mkdir $dumpdir
  preprocess.py --in-dir $data --out-dir $dumpdir --sample-rate $sample_rate
fi


expdir=/content/drive/'My Drive'/'Colab Notebooks'/Research_Test/exp/train_${tag}_r${sample_rate}_N${N}_L${L}_C${C}_${norm_type}_causal${causal}_${mask_nonlinear}_epoch${epochs}_half${half_lr}_norm${max_norm}_bs${batch_size}_worker${num_workers}_${optimizer}_lr${lr}_mmt${momentum}_rnn_b_layer${rnn_b_layer}_segment${segment}_l2${l2}_lw${lw}_tflip${tflip}_loss_every${loss_every}_lr_decay${lr_decay}_`basename $train_dir`
echo $expdir

if [ $stage -le 2 ]; then
  echo "Stage 2: Training"
  ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES="$id" \
    train_mloss.py \
    --train_dir $train_dir \
    --valid_dir $valid_dir \
    --sample_rate $sample_rate \
    --segment $segment \
    --cv_maxlen $cv_maxlen \
    --N $N \
    --L $L \
    --B $B \
    --H $H \
    --P $P \
    --X $X \
    --R $R \
    --C $C \
    --norm_type $norm_type \
    --causal $causal \
    --mask_nonlinear $mask_nonlinear \
    --use_cuda $use_cuda \
    --epochs $epochs \
    --half_lr $half_lr \
    --early_stop $early_stop \
    --max_norm $max_norm \
    --shuffle $shuffle \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --optimizer $optimizer \
    --lr $lr \
    --momentum $momentum \
    --l2 $l2 \
    --save_folder ${expdir} \
    --checkpoint $checkpoint \
    --continue_from "$continue_from" \
    --print_freq ${print_freq} \
    --visdom $visdom \
    --visdom_epoch $visdom_epoch \
    --visdom_id "$visdom_id" \
    --data_dir $evaluate_dir \
    --cal_sdr $cal_sdr \
    --batch_size_eval $batch_size_eval \
    --eval_every $eval_every \
    --lr_decay $lr_decay \
    --rnn_hidden_dim $rnn_hidden_dim \
    --lw $lw \
    --rnn_b_layer $rnn_b_layer \
    --tflip $tflip \
    --loss_every $loss_every \
    --ide $ide
fi
