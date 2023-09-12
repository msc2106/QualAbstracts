To open ssh (project and zone set at config)
```bash
gcloud compute ssh "instance-1" "--" -L 8888:localhost:8888
```

Starting colab server from inside ssh:
```bash
 jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
```
To start a normal server, only the port specification is needed.

Uploading and downloading from google cloud
```bash
gcloud storage cp {from} gs://qual_abstracts_data_transfer/{folder}
gcloud storage cp gs://{bucket}/{file_path} {to}
```
Note that this requires enabling Google service APIs when creating the VM.

Might need this:
```bash
jupyter nbextension enable --py widgetsnbextension
```

Resource monitoring:
```bash
df -h #disk space
ps a #processes
nvidia-smi #gpu
```

