docker  run \
    --name=squad_data_dl \
    -v ./data:/data:rw \
    --rm \
    deh_measurement \
    squad_qa_dl.py --cache_folder /data/qa_dl_cache/ --context_folder /data/contexts --qas_file /data/qas/squad_qas.tsv