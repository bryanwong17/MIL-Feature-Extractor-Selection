# Define variables
DEVICE="cuda:0"
DATASET="tcga-nsclc"
NUM_CLASSES=2
FEATURE_EXTRACTOR="resnet50-supervised-imagenet1k"
FEATS_SIZE=1024
SEEDS=(1 17 2000)

# Run training for each seed
for SEED in "${SEEDS[@]}"
do
    echo "Running ABMIL with seed $SEED"
    python train_abmil_dsmil.py --seed $SEED --device "$DEVICE" --num_classes $NUM_CLASSES --dataset "$DATASET" --feature_extractor "$FEATURE_EXTRACTOR" --feats_size $FEATS_SIZE --model abmil

    echo "Running DSMIL with seed $SEED"
    python train_abmil_dsmil.py --seed $SEED --device "$DEVICE" --num_classes $NUM_CLASSES --dataset "$DATASET" --feature_extractor "$FEATURE_EXTRACTOR" --feats_size $FEATS_SIZE --model dsmil

    echo "Running TransMIL with seed $SEED"
    python train_transmil.py --seed $SEED --device "$DEVICE" --num_classes $NUM_CLASSES --dataset "$DATASET" --feature_extractor "$FEATURE_EXTRACTOR" --feats_size $FEATS_SIZE --model transmil

    echo "Running DTFD-MIL (MaxMins) with seed $SEED"
    python train_dtfdmil.py --seed $SEED --device "$DEVICE" --num_classes $NUM_CLASSES --dataset "$DATASET" --feature_extractor "$FEATURE_EXTRACTOR" --feats_size $FEATS_SIZE --model DTFD --distill MaxMinS
done
