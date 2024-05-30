set -e

jupyter nbconvert --to notebook --execute feature_pipeline.ipynb

jupyter nbconvert --to notebook --execute predictions_pipeline.ipynb