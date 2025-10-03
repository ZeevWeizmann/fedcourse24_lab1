echo "=> Generate data.."

cd data/ || exit

rm -r mnist/all_data

/opt/homebrew/bin/python3  generate_data.py \
  --dataset mnist \
  --n_clients 10 \
  --iid \
  --frac 0.2 \
  --save_dir mnist \
  --seed 1234

cd ../

echo "=> Train.."



# Run FedAvg for different numbers of local epochs (e.g., 1, 5, 10, 50, 100)
for steps in 1 5 10 50 100
do
  echo "=> Train with local_steps=$steps, batch=128"
  python train.py \
    --experiment "mnist" \
    --n_rounds 100 \
    --local_steps $steps \
    --local_optimizer sgd \
    --local_lr 0.001 \
    --server_optimizer sgd \
    --server_lr 0.1 \
    --bz 128 \
    --device "cpu" \
    --log_freq 1 \
    --verbose 1 \
    --logs_dir "logs/mnist_steps_${steps}/" \
    --seed 12
done

# Bonus: Set the batch size equal to the dataset size, and the number of local epochs to 1.
echo "=> Train FedSGD (local_steps=1, bz=60000)"
python train.py \
  --experiment "mnist" \
  --n_rounds 100 \
  --local_steps 1 \
  --local_optimizer sgd \
  --local_lr 0.1 \
  --server_optimizer sgd \
  --server_lr 1 \
  --bz 60000 \
  --device "cpu" \
  --log_freq 1 \
  --verbose 1 \
  --logs_dir "logs/mnist_FedSGD/" \
  --seed 12
