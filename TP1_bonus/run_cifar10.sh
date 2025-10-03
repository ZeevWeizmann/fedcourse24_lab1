echo "=> Train on CIFAR10 with MobileNet.."

python data/generate_data.py \
  --dataset_name cifar10 \
  --n_clients 10 \
  --iid \
  --frac 1.0 \
  --save_dir data/cifar10 \
  --seed 123

# Run FedAvg for different hyperparameter setups
for steps in 1 5 10
do
  for lr in 0.01 0.001
  do
    echo "=> Train with local_steps=$steps, lr=$lr, batch=128"
    python train.py \
      --experiment "cifar10" \
      --n_rounds 50 \
      --local_steps $steps \
      --local_optimizer sgd \
      --local_lr $lr \
      --server_optimizer sgd \
      --server_lr 0.1 \
      --bz 128 \
      --device "cpu" \
      --log_freq 1 \
      --verbose 1 \
      --logs_dir "logs/cifar10_steps_${steps}_lr_${lr}/" \
      --seed 42
  done
done