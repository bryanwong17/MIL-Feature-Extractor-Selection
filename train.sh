python train_abmil_dsmil.py --seed 0 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model abmil
python train_abmil_dsmil.py --seed 0 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model dsmil
python train_transmil.py --seed 0 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model transmil
python train_dtfdmil.py --seed 0 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model DTFD --distill MaxMinS

python train_abmil_dsmil.py --seed 5 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model abmil
python train_abmil_dsmil.py --seed 5 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model dsmil
python train_transmil.py --seed 5 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model transmil
python train_dtfdmil.py --seed 5 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model DTFD --distill MaxMinS

python train_abmil_dsmil.py --seed 10 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model abmil
python train_abmil_dsmil.py --seed 10 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model dsmil
python train_transmil.py --seed 10 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model transmil
python train_dtfdmil.py --seed 10 --device cuda:0 --num_classes 2 --dataset tcga-nsclc --feature_extractor resnet50-supervised-imagenet1k-transform --feats_size 1024 --model DTFD --distill MaxMinS