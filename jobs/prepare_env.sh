module load python/base/miniconda3
conda create --name myenv pytorch torchvision tensorboard jupyterlab pandas numpy matplotlib scikit-learn scipy cudatoolkit=10.2 --c pytorch -c conda-forge
source /apps/sw/miniconda/etc/profile.d/conda.sh
