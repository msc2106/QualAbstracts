To open ssh (project and zone set at config)
```bash
gcloud compute ssh "instance-1" "--" -L 8888:localhost:8888
```

Starting server from inside ssh:
```bash
 jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
```

Uploading and downloading from google cloud
```bash
gcloud storage cp {from} gs://{bucket}/
gcloud storage cp gs://{bucket}/{file_path} {to}
```

Might need this:
```bash
jupyter nbextension enable --py widgetsnbextension
```