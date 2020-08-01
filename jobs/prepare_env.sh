module load python/base/miniconda3
conda create --name myenv pytorch torchvision tensorboard pandas numpy matplotlib scikit-learn scipy --channel pytorch
source /apps/sw/miniconda/etc/profile.d/conda.sh
