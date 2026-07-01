# PyTorch training environment.
# Runs the 1D CNN inside a Linux container so Windows Smart App Control (which
# blocks the native torch.dll on the host) does not apply.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /work
RUN pip install --no-cache-dir pyarrow scikit-learn pandas

# Code and data are bind-mounted at run time (see docker-train.sh), so the
# 15 GB of parquet never gets baked into the image.
CMD ["python", "code/train_cnn.py"]
